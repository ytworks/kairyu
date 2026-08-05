"""CPU-only contract and tamper tests for the KV equivalence sidecar."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kairyu.bench import kv_equivalence as gate


def _prompt(cell_id: str, suffix: str = "main", *, token_count: int = 1024) -> dict[str, object]:
    cell_offset = 20_000 * gate.CELL_IDS.index(cell_id)
    suffix_offset = (
        0 if suffix == "main" else 1_000 + int(gate.sha256_bytes(suffix.encode())[:5], 16) % 9_000
    )
    token_ids = [cell_offset + suffix_offset + index for index in range(token_count)]
    return {
        "token_ids": token_ids,
        "token_ids_sha256": gate.sha256_json(token_ids),
        "token_count": token_count,
    }


def _response(
    cell_id: str,
    suffix: str,
    prompt: MappingLike,
    *,
    cached_tokens: int,
) -> dict[str, object]:
    offset = 1_000 * gate.CELL_IDS.index(cell_id) + (100 if suffix != "answer" else 0)
    token_ids = [offset + index for index in range(gate.OUTPUT_TOKENS)]
    token_pieces = [f"<{token}>" for token in token_ids]
    return {
        "status": "success",
        "token_ids": token_ids,
        "token_pieces": token_pieces,
        "text": "".join(token_pieces),
        "finish_reason": "length",
        "usage": {
            "prompt_tokens": prompt["token_count"],
            "completion_tokens": gate.OUTPUT_TOKENS,
            "cached_tokens": cached_tokens,
        },
    }


MappingLike = dict[str, object]


def _source() -> dict[str, object]:
    files = {path: gate.sha256_json(["source", path]) for path in gate.REQUIRED_SOURCE_PATHS}
    return {
        "git_head": "1" * 40,
        "tracked_tree_clean": True,
        "file_sha256": files,
        "files_sha256": gate.sha256_json(files),
    }


def _checkpoint() -> dict[str, object]:
    return {
        "model": gate.MODEL,
        "revision": gate.MODEL_REVISION,
        "architecture": copy.deepcopy(gate.EXPECTED_ARCHITECTURE),
        "weights": copy.deepcopy(list(gate.EXPECTED_WEIGHTS)),
        "weights_sha256": gate.EXPECTED_WEIGHTS_SHA256,
        "pinned_weights_rollup": gate.EXPECTED_WEIGHTS_ROLLUP,
        "metadata_sha256": copy.deepcopy(gate.EXPECTED_METADATA_SHA256),
        "weight_index_sha256": gate.EXPECTED_WEIGHT_INDEX_SHA256,
        "model_config_sha256": gate.EXPECTED_MODEL_CONFIG_SHA256,
    }


def _checkpoint_binding(cell_id: str) -> dict[str, object]:
    kind = gate.CELL_SPECS[cell_id]["parent_binding_kind"]
    if kind == "direct-partial":
        return {
            "binding_kind": kind,
            "model": gate.MODEL,
            "revision": gate.MODEL_REVISION,
            "pinned_weights_rollup": gate.EXPECTED_WEIGHTS_ROLLUP,
        }
    if kind == "aggregate-common-f2c":
        return {"binding_kind": kind}
    checkpoint = _checkpoint()
    checkpoint.pop("weight_index_sha256")
    return {"binding_kind": kind, **checkpoint}


def _parent(cell_id: str) -> dict[str, object]:
    spec = gate.CELL_SPECS[cell_id]
    manifest_identity = "f4a" if cell_id.startswith("f4a-") else cell_id
    return {
        "gate": spec["parent_gate"],
        "schema_version": spec["parent_schema_version"],
        "manifest_sha256": gate.sha256_json([manifest_identity, "manifest"]),
        "raw_sha256": gate.sha256_json([cell_id, "raw"]),
        "quality_manifest_sha256": (
            gate.sha256_json([cell_id, "quality-manifest"]) if cell_id == "f4b-tp4" else None
        ),
        "quality_raw_sha256": (
            gate.sha256_json([cell_id, "quality-raw"]) if cell_id == "f4b-tp4" else None
        ),
        "parent_topology": copy.deepcopy(spec["parent_topology"]),
        "checkpoint_binding": _checkpoint_binding(cell_id),
    }


def _engine(cell_id: str) -> dict[str, object]:
    spec = gate.CELL_SPECS[cell_id]
    return {
        "cell_id": cell_id,
        "tensor_parallel_size": spec["tensor_parallel_size"],
        "page_size": spec["page_size"],
        "num_pages": spec["num_pages"],
        "dram_profile_sha256": spec["dram_profile_sha256"],
        "dram_capacity_pages": spec["dram_capacity_pages"],
        "dram_min_restore_tokens": spec["dram_min_restore_tokens"],
        "decode_mode": spec["decode_mode"],
        "max_num_batched_tokens": spec["max_num_batched_tokens"],
        "max_model_len": spec["max_model_len"],
    }


def _runtime(cell_id: str) -> dict[str, object]:
    return {
        "run_nonce": f"{gate.CELL_IDS.index(cell_id) + 1:032x}",
        "source": _source(),
        "checkpoint": _checkpoint(),
        "engine": _engine(cell_id),
    }


def _request(
    run_id: str,
    cell_id: str,
    phase: str,
    prompt: MappingLike,
    runtime_nonce: str,
    *,
    suffix: str,
    cached_tokens: int,
) -> dict[str, object]:
    feature = gate.CELL_SPECS[cell_id]["feature_id"]
    return {
        "row_type": "request",
        "run_id": run_id,
        "cell_id": cell_id,
        "feature_id": feature,
        "phase": phase,
        "runtime_nonce": runtime_nonce,
        "prompt": copy.deepcopy(prompt),
        "response": _response(cell_id, suffix, prompt, cached_tokens=cached_tokens),
        "error_type": None,
        "retry_count": 0,
    }


def _stats(
    *,
    offload_pages: int = 0,
    offload_bypassed_pages: int = 0,
    restore_pages: int = 0,
    restore_attempts: int = 0,
    restore_fallbacks: int = 0,
    ownership_failures: int = 0,
) -> dict[str, int]:
    return {
        "offload_pages": offload_pages,
        "offload_bypassed_pages": offload_bypassed_pages,
        "restore_pages": restore_pages,
        "restore_attempts": restore_attempts,
        "restore_fallbacks": restore_fallbacks,
        "ownership_failures": ownership_failures,
    }


def _feature_proof(cell_id: str, prompt: MappingLike) -> dict[str, object]:
    if gate.CELL_SPECS[cell_id]["transition"] == "radix":
        return {
            "kind": "radix_reuse",
            "same_runtime": True,
            "cache_hit_tokens": prompt["token_count"],
        }
    return {
        "kind": "dram_restore",
        "before": _stats(),
        "pre_warm": _stats(offload_pages=70),
        "after": _stats(offload_pages=70, restore_pages=64, restore_attempts=1),
        "pressure_delta": _stats(offload_pages=70),
        "warm_delta": _stats(restore_pages=64, restore_attempts=1),
        "target_cached_tokens_pre_warm": 0,
    }


def _cell_rows(cell_id: str, *, run_id: str = "cpu-fixture-373") -> list[dict[str, object]]:
    spec = gate.CELL_SPECS[cell_id]
    feature = spec["feature_id"]
    main = _prompt(cell_id)
    runtime = _runtime(cell_id)
    nonce = str(runtime["run_nonce"])
    rows: list[dict[str, object]] = [
        {
            "row_type": "cell_start",
            "run_id": run_id,
            "cell_id": cell_id,
            "feature_id": feature,
            "transition": spec["transition"],
            "prompt": copy.deepcopy(main),
            "sampling": gate.frozen_config()["sampling"],
            "parent_evidence": _parent(cell_id),
            "runtime": runtime,
        },
        _request(
            run_id,
            cell_id,
            "cold",
            main,
            nonce,
            suffix="answer",
            cached_tokens=0,
        ),
    ]
    if spec["transition"] == "dram_restore":
        for pressure_index in range(2):
            pressure = _prompt(cell_id, f"pressure-{pressure_index}")
            rows.append(
                _request(
                    run_id,
                    cell_id,
                    "pressure",
                    pressure,
                    nonce,
                    suffix=f"pressure-{pressure_index}",
                    cached_tokens=0,
                )
            )
    rows.extend(
        [
            _request(
                run_id,
                cell_id,
                "warm",
                main,
                nonce,
                suffix="answer",
                cached_tokens=int(main["token_count"]),
            ),
            {
                "row_type": "cell_end",
                "run_id": run_id,
                "cell_id": cell_id,
                "feature_id": feature,
                "feature_proof": _feature_proof(cell_id, main),
            },
        ]
    )
    return rows


def _rows() -> list[dict[str, object]]:
    run_id = "cpu-fixture-373"
    rows: list[dict[str, object]] = [
        {
            "row_type": "run",
            "schema_version": gate.SCHEMA_VERSION,
            "gate": gate.GATE,
            "run_id": run_id,
            "required_cells": list(gate.CELL_IDS),
            "config": gate.frozen_config(),
        }
    ]
    for cell_id in gate.CELL_IDS:
        rows.extend(_cell_rows(cell_id, run_id=run_id))
    rows.append(
        {
            "row_type": "run_end",
            "run_id": run_id,
            "status": "complete",
            "errors": [],
            "cell_count": len(gate.CELL_IDS),
        }
    )
    return rows


def _row(
    rows: list[dict[str, object]],
    cell_id: str,
    row_type: str,
    phase: str | None = None,
    occurrence: int = 0,
) -> dict[str, object]:
    candidates = [
        row
        for row in rows
        if row.get("cell_id") == cell_id
        and row.get("row_type") == row_type
        and (phase is None or row.get("phase") == phase)
    ]
    return candidates[occurrence]


def _cell_slice(rows: list[dict[str, object]], cell_id: str) -> list[dict[str, object]]:
    start = rows.index(_row(rows, cell_id, "cell_start"))
    end = rows.index(_row(rows, cell_id, "cell_end"))
    return rows[start : end + 1]


def _manifest(rows: list[dict[str, object]]) -> dict[str, object]:
    return gate.recompute_manifest(rows, raw_sha256="a" * 64)


def test_public_contract_freezes_five_production_topology_cells() -> None:
    assert gate.FEATURES == ("f2c", "f2d", "f4a", "f4b")
    assert gate.CELL_IDS == (
        "f2c-tp2",
        "f2d-tp2",
        "f4a-tp4",
        "f4a-tp8",
        "f4b-tp4",
    )
    assert [gate.CELL_SPECS[cell]["tensor_parallel_size"] for cell in gate.CELL_IDS] == [
        2,
        2,
        4,
        8,
        4,
    ]
    assert gate.CELL_SPECS["f2c-tp2"]["parent_topology"] == {
        "kind": "direct",
        "tensor_parallel_size": 2,
        "service_count": 4,
        "service_device_ids": {
            "replica-a0": ["0", "1"],
            "replica-a1": ["2", "3"],
            "replica-b0": ["4", "5"],
            "replica-b1": ["6", "7"],
        },
        "unique_device_count": 8,
    }
    assert gate.CELL_SPECS["f4b-tp4"]["dram_capacity_pages"] == 2048
    assert gate.CELL_SPECS["f4a-tp8"]["max_model_len"] == 8193
    assert gate.frozen_config()["sampling"]["skip_special_tokens"] is False


@pytest.mark.parametrize("cell_id", gate.CELL_IDS)
def test_pure_single_cell_validator_accepts_each_fixed_cell(cell_id: str) -> None:
    cell = gate.validate_single_cell(_cell_rows(cell_id), expected_cell_id=cell_id)

    assert cell["cell_id"] == cell_id
    assert cell["passed"] is True
    assert all(cell["checks"].values())


def test_valid_fixture_recomputes_complete_passing_manifest() -> None:
    rows = _rows()
    manifest = _manifest(rows)

    assert manifest["passed"] is True
    assert [cell["cell_id"] for cell in manifest["cells"]] == list(gate.CELL_IDS)
    assert manifest["checks"]["five_cells_exact_order_and_production_topology"] is True
    assert manifest["checks"]["all_runtime_checkpoint_identities_exact"] is True
    assert manifest["checks"]["f2d_runtime_checkpoint_equals_f2c"] is True
    assert manifest["checks"]["f4a_tp4_tp8_share_manifest_and_use_distinct_raw_shards"] is True
    assert manifest["checks"]["every_cell_uses_a_unique_runtime_nonce"] is True
    gate.assert_gate(manifest)


def test_canonical_artifact_round_trip_never_trusts_stored_passed(tmp_path: Path) -> None:
    raw = tmp_path / gate.RAW_NAME
    retained = tmp_path / gate.MANIFEST_NAME
    written = gate.write_artifact(raw, retained, _rows(), assert_gate=True)

    assert gate.read_jsonl(raw) == _rows()
    assert gate.replay_raw(raw, assert_gate=True) == written
    assert gate.verify_artifact(tmp_path, assert_gate=True) == written
    assert retained.read_text(encoding="utf-8") == gate.canonical_json(written) + "\n"


@pytest.mark.parametrize("surface", ["token_ids", "token_pieces", "text"])
def test_native_answer_tamper_fails_single_cell_and_aggregate(surface: str) -> None:
    rows = _rows()
    warm = _row(rows, "f2c-tp2", "request", "warm")["response"]
    if surface == "token_ids":
        warm[surface][0] += 1
    elif surface == "token_pieces":
        warm[surface][0] += "tampered"
    else:
        warm[surface] += "tampered"

    child = gate.validate_single_cell(_cell_slice(rows, "f2c-tp2"))
    aggregate = _manifest(rows)

    check = "f2c-tp2_cold_and_warm_native_answer_exact"
    assert child["checks"][check] is False
    assert child["passed"] is False
    assert aggregate["checks"][check] is False
    assert aggregate["passed"] is False


def test_fake_small_checkpoint_is_rejected() -> None:
    rows = _cell_rows("f2d-tp2")
    checkpoint = rows[0]["runtime"]["checkpoint"]
    checkpoint["weights"] = checkpoint["weights"][:1]
    checkpoint["weights_sha256"] = gate.sha256_json(checkpoint["weights"])
    checkpoint["pinned_weights_rollup"] = gate.legacy_weights_rollup(checkpoint["weights"])

    with pytest.raises(gate.EvidenceError, match="17 shards"):
        gate.validate_single_cell(rows)


@pytest.mark.parametrize("surface", ["metadata", "weight_index", "model_config"])
def test_checkpoint_metadata_identity_is_pinned(surface: str) -> None:
    rows = _cell_rows("f2d-tp2")
    checkpoint = rows[0]["runtime"]["checkpoint"]
    if surface == "metadata":
        checkpoint["metadata_sha256"]["tokenizer.json"] = "f" * 64
    elif surface == "weight_index":
        checkpoint["weight_index_sha256"] = "f" * 64
    else:
        checkpoint["model_config_sha256"] = "f" * 64

    with pytest.raises(gate.EvidenceError):
        gate.validate_single_cell(rows)


@pytest.mark.parametrize(
    ("cell_id", "field", "value"),
    [
        ("f2c-tp2", "tensor_parallel_size", 1),
        ("f2c-tp2", "num_pages", 128),
        ("f2d-tp2", "decode_mode", "eager"),
        ("f4a-tp4", "dram_capacity_pages", 1),
        ("f4a-tp8", "max_model_len", 8192),
        ("f4b-tp4", "max_num_batched_tokens", 1024),
    ],
)
def test_runtime_engine_rejects_nonproduction_geometry(
    cell_id: str,
    field: str,
    value: object,
) -> None:
    rows = _cell_rows(cell_id)
    rows[0]["runtime"]["engine"][field] = value

    with pytest.raises(gate.EvidenceError, match=field):
        gate.validate_single_cell(rows)


def test_missing_f4a_tp8_cell_is_rejected() -> None:
    rows = _rows()
    for row in list(_cell_slice(rows, "f4a-tp8")):
        rows.remove(row)

    with pytest.raises(gate.EvidenceError):
        _manifest(rows)


@pytest.mark.parametrize("surface", ["clean", "paths", "rollup", "git_head"])
def test_dirty_or_incomplete_source_is_rejected(surface: str) -> None:
    rows = _cell_rows("f2c-tp2")
    source = rows[0]["runtime"]["source"]
    if surface == "clean":
        source["tracked_tree_clean"] = False
    elif surface == "paths":
        source["file_sha256"].pop(next(iter(source["file_sha256"])))
        source["files_sha256"] = gate.sha256_json(source["file_sha256"])
    elif surface == "rollup":
        source["files_sha256"] = "f" * 64
    else:
        source["git_head"] = "not-a-commit"

    with pytest.raises(gate.EvidenceError):
        gate.validate_single_cell(rows)


def test_aggregate_requires_one_exact_clean_source_identity() -> None:
    rows = _rows()
    source = _row(rows, "f2d-tp2", "cell_start")["runtime"]["source"]
    path = gate.REQUIRED_SOURCE_PATHS[0]
    source["file_sha256"][path] = "f" * 64
    source["files_sha256"] = gate.sha256_json(source["file_sha256"])

    manifest = _manifest(rows)

    assert manifest["checks"]["all_cells_use_one_clean_source_identity"] is False
    assert manifest["passed"] is False


@pytest.mark.parametrize("cell_id", ["f2c-tp2", "f4a-tp4", "f4b-tp4"])
def test_parent_checkpoint_binding_mismatch_is_rejected(cell_id: str) -> None:
    rows = _cell_rows(cell_id)
    binding = rows[0]["parent_evidence"]["checkpoint_binding"]
    binding["revision"] = "2" * 40

    with pytest.raises(gate.EvidenceError, match="revision"):
        gate.validate_single_cell(rows)


def test_f2c_parent_topology_must_bind_four_tp2_services_and_eight_unique_gpus() -> None:
    rows = _cell_rows("f2c-tp2")
    topology = rows[0]["parent_evidence"]["parent_topology"]
    topology["service_device_ids"]["replica-b1"] = ["0", "1"]

    with pytest.raises(gate.EvidenceError, match="topology"):
        gate.validate_single_cell(rows)


def test_f4a_requires_one_parent_manifest_and_two_distinct_raw_shards() -> None:
    rows = _rows()
    tp8 = _row(rows, "f4a-tp8", "cell_start")["parent_evidence"]
    tp4 = _row(rows, "f4a-tp4", "cell_start")["parent_evidence"]
    tp8["raw_sha256"] = tp4["raw_sha256"]

    manifest = _manifest(rows)

    check = "f4a_tp4_tp8_share_manifest_and_use_distinct_raw_shards"
    assert manifest["checks"][check] is False
    assert manifest["passed"] is False


def test_runtime_nonce_is_unique_per_cell_and_bound_to_every_request() -> None:
    rows = _rows()
    f2c_nonce = _row(rows, "f2c-tp2", "cell_start")["runtime"]["run_nonce"]
    f2d = _row(rows, "f2d-tp2", "cell_start")
    f2d["runtime"]["run_nonce"] = f2c_nonce
    requests = [
        row for row in rows if row.get("cell_id") == "f2d-tp2" and row.get("row_type") == "request"
    ]
    for request in requests:
        request["runtime_nonce"] = f2c_nonce

    manifest = _manifest(rows)
    assert manifest["checks"]["every_cell_uses_a_unique_runtime_nonce"] is False
    assert manifest["passed"] is False

    rows = _cell_rows("f2c-tp2")
    rows[1]["runtime_nonce"] = "f" * 32
    with pytest.raises(gate.EvidenceError, match="runtime nonce"):
        gate.validate_single_cell(rows)


@pytest.mark.parametrize(
    "mutation",
    ["pressure_error", "pressure_cached", "fallback", "target_not_evicted", "no_restore"],
)
def test_pressure_or_restore_failure_is_consistently_rejected(mutation: str) -> None:
    rows = _rows()
    pressure = _row(rows, "f4a-tp4", "request", "pressure")
    proof = _row(rows, "f4a-tp4", "cell_end")["feature_proof"]
    if mutation == "pressure_error":
        pressure["response"]["status"] = "error"
        pressure["error_type"] = "RuntimeError"
    elif mutation == "pressure_cached":
        pressure["response"]["usage"]["cached_tokens"] = 1
    elif mutation == "fallback":
        proof["after"]["restore_fallbacks"] = 1
        proof["warm_delta"]["restore_fallbacks"] = 1
    elif mutation == "target_not_evicted":
        proof["target_cached_tokens_pre_warm"] = 16
    else:
        proof["after"]["restore_pages"] = 0
        proof["after"]["restore_attempts"] = 0
        proof["warm_delta"]["restore_pages"] = 0
        proof["warm_delta"]["restore_attempts"] = 0

    child = gate.validate_single_cell(_cell_slice(rows, "f4a-tp4"))
    aggregate = _manifest(rows)

    assert child["passed"] is False
    assert aggregate["passed"] is False


@pytest.mark.parametrize("cell_id", ["f4a-tp4", "f4b-tp4"])
def test_dram_restore_must_meet_frozen_minimum(cell_id: str) -> None:
    rows = _rows()
    proof = _row(rows, cell_id, "cell_end")["feature_proof"]
    proof["after"]["restore_pages"] = 1
    proof["warm_delta"]["restore_pages"] = 1

    child = gate.validate_single_cell(_cell_slice(rows, cell_id))
    aggregate = _manifest(rows)

    assert child["passed"] is False
    assert aggregate["passed"] is False


def test_tp8_one_page_restore_meets_its_frozen_minimum() -> None:
    rows = _rows()
    proof = _row(rows, "f4a-tp8", "cell_end")["feature_proof"]
    proof["after"]["restore_pages"] = 1
    proof["warm_delta"]["restore_pages"] = 1

    assert gate.validate_single_cell(_cell_slice(rows, "f4a-tp8"))["passed"] is True
    assert _manifest(rows)["passed"] is True


def test_pressure_answers_are_excluded_from_cold_warm_equality() -> None:
    rows = _rows()
    pressure = _row(rows, "f4b-tp4", "request", "pressure")
    pressure["response"]["token_ids"] = list(reversed(pressure["response"]["token_ids"]))
    pressure["response"]["token_pieces"] = [
        f"unrelated-{index}" for index in range(gate.OUTPUT_TOKENS)
    ]
    pressure["response"]["text"] = "unrelated pressure answer"

    assert gate.validate_single_cell(_cell_slice(rows, "f4b-tp4"))["passed"] is True
    assert _manifest(rows)["passed"] is True


def test_dram_prompt_below_1024_token_floor_is_rejected() -> None:
    rows = _cell_rows("f4a-tp8")
    for row in rows:
        if "prompt" not in row:
            continue
        prompt = row["prompt"]
        prompt["token_ids"] = prompt["token_ids"][:512]
        prompt["token_count"] = 512
        prompt["token_ids_sha256"] = gate.sha256_json(prompt["token_ids"])
        if row.get("row_type") == "request":
            row["response"]["usage"]["prompt_tokens"] = 512
            if row.get("phase") == "warm":
                row["response"]["usage"]["cached_tokens"] = 512

    with pytest.raises(gate.EvidenceError, match="exactly 1024 tokens"):
        gate.validate_single_cell(rows)


def test_radix_comparison_prompt_cannot_shrink_below_fixed_1024_tokens() -> None:
    rows = _cell_rows("f2c-tp2")
    for row in rows:
        if "prompt" not in row:
            continue
        prompt = row["prompt"]
        prompt["token_ids"] = prompt["token_ids"][:512]
        prompt["token_count"] = 512
        prompt["token_ids_sha256"] = gate.sha256_json(prompt["token_ids"])
        if row.get("row_type") == "request":
            row["response"]["usage"]["prompt_tokens"] = 512
            if row.get("phase") == "warm":
                row["response"]["usage"]["cached_tokens"] = 512
    rows[-1]["feature_proof"]["cache_hit_tokens"] = 512

    with pytest.raises(gate.EvidenceError, match="exactly 1024 tokens"):
        gate.validate_single_cell(rows)


def test_exact_cell_order_and_required_cells_are_structural() -> None:
    rows = _rows()
    rows[0]["required_cells"] = list(reversed(gate.CELL_IDS))
    with pytest.raises(gate.EvidenceError, match="required cell order"):
        _manifest(rows)

    rows = _rows()
    _row(rows, "f2c-tp2", "cell_start")["cell_id"] = "f2d-tp2"
    with pytest.raises(gate.EvidenceError):
        _manifest(rows)


def test_f4b_parent_requires_sealed_quality_manifest_and_raw_bindings() -> None:
    rows = _cell_rows("f4b-tp4")
    rows[0]["parent_evidence"]["quality_raw_sha256"] = None

    with pytest.raises(gate.EvidenceError, match="quality raw digest"):
        gate.validate_single_cell(rows)

    rows = _cell_rows("f2c-tp2")
    rows[0]["parent_evidence"]["quality_manifest_sha256"] = "a" * 64
    with pytest.raises(gate.EvidenceError, match="unexpected quality evidence"):
        gate.validate_single_cell(rows)


@pytest.mark.parametrize("surface", ["token_ids", "token_ids_sha256", "token_count"])
def test_prompt_descriptor_is_recomputed_from_retained_ids(surface: str) -> None:
    rows = _cell_rows("f2c-tp2")
    prompt = rows[1]["prompt"]
    if surface == "token_ids":
        prompt["token_ids"][0] += 1
    elif surface == "token_ids_sha256":
        prompt["token_ids_sha256"] = "f" * 64
    else:
        prompt["token_count"] += 1

    with pytest.raises(gate.EvidenceError, match="raw ids"):
        gate.validate_single_cell(rows)


@pytest.mark.parametrize("surface", ["prompt", "output"])
def test_replay_rejects_token_ids_outside_the_pinned_vocabulary(surface: str) -> None:
    rows = _cell_rows("f2c-tp2")
    if surface == "prompt":
        for row in rows:
            if "prompt" not in row:
                continue
            row["prompt"]["token_ids"][0] = gate.TOKENIZER_VOCAB_SIZE
            row["prompt"]["token_ids_sha256"] = gate.sha256_json(row["prompt"]["token_ids"])
    else:
        rows[1]["response"]["token_ids"][0] = gate.TOKENIZER_VOCAB_SIZE

    with pytest.raises(gate.EvidenceError, match="token ids are invalid"):
        gate.validate_single_cell(rows)


def test_replay_accepts_the_highest_tokenizer_owned_id() -> None:
    rows = _cell_rows("f2c-tp2")
    highest = gate.TOKENIZER_VOCAB_SIZE - 1
    for row in rows:
        if "prompt" in row:
            row["prompt"]["token_ids"] = [highest] * row["prompt"]["token_count"]
            row["prompt"]["token_ids_sha256"] = gate.sha256_json(row["prompt"]["token_ids"])
        if row.get("row_type") == "request":
            row["response"]["token_ids"] = [highest] * gate.OUTPUT_TOKENS

    assert gate.validate_single_cell(rows)["passed"] is True


def test_replay_rejects_empty_native_token_piece() -> None:
    rows = _cell_rows("f2c-tp2")
    rows[1]["response"]["token_pieces"][0] = ""

    with pytest.raises(gate.EvidenceError, match="token pieces are invalid"):
        gate.validate_single_cell(rows)


@pytest.mark.parametrize(
    "surface",
    ["sampling", "engine", "architecture", "parent_topology", "config", "cell_count"],
)
def test_replay_rejects_json_numeric_type_confusion(surface: str) -> None:
    rows = _rows()
    start = _row(rows, "f2c-tp2", "cell_start")
    if surface == "sampling":
        start["sampling"]["temperature"] = False
    elif surface == "engine":
        start["runtime"]["engine"]["page_size"] = 16.0
    elif surface == "architecture":
        start["runtime"]["checkpoint"]["architecture"]["hidden_size"] = 5120.0
    elif surface == "parent_topology":
        start["parent_evidence"]["parent_topology"]["unique_device_count"] = 8.0
    elif surface == "config":
        rows[0]["config"]["output_tokens"] = 32.0
    else:
        rows[-1]["cell_count"] = 5.0

    with pytest.raises(gate.EvidenceError):
        _manifest(rows)


@pytest.mark.parametrize(
    "payload",
    ['{"a":1}', '{"a": 1}\n', '{"a":1,"a":1}\n', '{"a":NaN}\n', "\n"],
)
def test_jsonl_reader_rejects_noncanonical_or_ambiguous_bytes(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(gate.EvidenceError):
        gate.read_jsonl(path)


def test_replay_hashes_the_same_immutable_snapshot_it_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / gate.RAW_NAME
    gate.write_jsonl(raw, _rows())
    original_bytes = raw.read_bytes()
    original_snapshot = gate._read_jsonl_snapshot

    def replace_after_snapshot(path: Path):
        snapshot = original_snapshot(path)
        path.write_text("{}\n", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(gate, "_read_jsonl_snapshot", replace_after_snapshot)
    manifest = gate.replay_raw(raw, assert_gate=True)

    assert manifest["raw"]["sha256"] == gate.sha256_bytes(original_bytes)
    monkeypatch.setattr(gate, "_read_jsonl_snapshot", original_snapshot)
    with pytest.raises(gate.EvidenceError):
        gate.replay_raw(raw, assert_gate=True)


def test_verify_rejects_forged_retained_passed_value(tmp_path: Path) -> None:
    rows = _rows()
    _row(rows, "f2c-tp2", "request", "warm")["response"]["text"] += "tampered"
    raw = tmp_path / gate.RAW_NAME
    retained = tmp_path / gate.MANIFEST_NAME
    manifest = gate.write_artifact(raw, retained, rows)
    forged = copy.deepcopy(manifest)
    forged["passed"] = True
    forged["checks"] = {name: True for name in forged["checks"]}
    retained.write_text(gate.canonical_json(forged) + "\n", encoding="utf-8")

    with pytest.raises(gate.EvidenceError, match="independent raw replay"):
        gate.verify_artifact(raw, retained)


def test_verify_rejects_json_type_confusion_in_retained_manifest(tmp_path: Path) -> None:
    raw = tmp_path / gate.RAW_NAME
    retained = tmp_path / gate.MANIFEST_NAME
    manifest = gate.write_artifact(raw, retained, _rows(), assert_gate=True)
    manifest["passed"] = 1
    retained.write_text(gate.canonical_json(manifest) + "\n", encoding="utf-8")

    with pytest.raises(gate.EvidenceError, match="independent raw replay"):
        gate.verify_artifact(raw, retained)


def test_retained_manifest_must_be_canonical(tmp_path: Path) -> None:
    raw = tmp_path / gate.RAW_NAME
    retained = tmp_path / gate.MANIFEST_NAME
    manifest = gate.write_artifact(raw, retained, _rows())
    retained.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(gate.EvidenceError, match="not canonical"):
        gate.verify_artifact(raw, retained)


@pytest.mark.parametrize("existing_name", [gate.RAW_NAME, gate.MANIFEST_NAME])
def test_write_artifact_never_overwrites_an_existing_target(
    tmp_path: Path, existing_name: str
) -> None:
    raw = tmp_path / gate.RAW_NAME
    retained = tmp_path / gate.MANIFEST_NAME
    existing = tmp_path / existing_name
    existing.write_bytes(b"owner-data\n")

    with pytest.raises(gate.EvidenceError, match="refusing to overwrite"):
        gate.write_artifact(raw, retained, _rows())

    assert existing.read_bytes() == b"owner-data\n"
    other = retained if existing == raw else raw
    assert not other.exists()


def test_write_artifact_rejects_a_target_replaced_before_post_write_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / gate.RAW_NAME
    retained = tmp_path / gate.MANIFEST_NAME
    original_read_bytes = Path.read_bytes

    def read_with_replacement(path: Path) -> bytes:
        if path == raw and path.exists():
            return b"replaced-after-exclusive-create\n"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_with_replacement)
    with pytest.raises(gate.EvidenceError, match="changed while writing"):
        gate.write_artifact(raw, retained, _rows())


def test_aggregate_rejects_non_object_rows_with_contract_error() -> None:
    rows: list[object] = list(_rows())
    rows[1] = ["not", "an", "object"]

    with pytest.raises(gate.EvidenceError, match="every raw row must be an object"):
        gate.recompute_manifest(rows, raw_sha256="a" * 64)  # type: ignore[arg-type]
