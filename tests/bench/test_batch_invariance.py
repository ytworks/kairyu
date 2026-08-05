"""CPU-only contract and tamper tests for the A12 formal gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kairyu.bench import batch_invariance as gate


def _source_snapshot() -> dict[str, object]:
    files = {path: gate.sha256_json(["source", path]) for path in gate.REQUIRED_SOURCE_PATHS}
    return {
        "git_head": "1" * 40,
        "tracked_tree_clean": True,
        "file_sha256": files,
        "files_sha256": gate.sha256_json(files),
        "python_isolated": True,
        "bytecode_disabled": True,
    }


def _source() -> dict[str, object]:
    snapshot = _source_snapshot()
    return {
        "measurement_start": snapshot,
        "measurement_end": copy.deepcopy(snapshot),
        "stable_across_measurement": True,
    }


def _checkpoint_snapshot() -> dict[str, object]:
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


def _checkpoint() -> dict[str, object]:
    snapshot = _checkpoint_snapshot()
    return {
        "measurement_start": snapshot,
        "measurement_end": copy.deepcopy(snapshot),
        "stable_across_measurement": True,
    }


def _uuid(rank: int) -> str:
    value = rank + 1
    return f"GPU-{value:08x}-0000-0000-0000-{value:012x}"


def _hardware() -> dict[str, object]:
    return {
        "gpu_count": gate.TENSOR_PARALLEL_SIZE,
        "gpu_exclusive": True,
        "gpus": [
            {
                "rank": rank,
                "visible_index": rank,
                "name": gate.GPU_NAME,
                "uuid": _uuid(rank),
                "pci_bus_id": f"00000000:{rank + 1:02X}:00.0",
                "compute_capability": gate.GPU_COMPUTE_CAPABILITY,
                "total_memory_bytes": 100_000_000_000,
            }
            for rank in range(gate.TENSOR_PARALLEL_SIZE)
        ],
        "software": {
            "python_version": "3.12.1",
            "torch_version": "2.9.0",
            "cuda_version": "13.0",
            "driver_version": "580.0",
            "nccl_version": "2.28.0",
            "flashinfer_version": "0.6.14",
        },
    }


def _runtime(runtime_id: str) -> dict[str, object]:
    attention_identity = gate.canonical_json(
        {
            "requested": "flashinfer",
            "resolved": "flashinfer",
            "source": "env",
            "components": {
                "prefill": "flashinfer",
                "decode": "flashinfer",
                "kv_mode": "paged-direct",
            },
            "component_versions": {
                "prefill": "0.6.14",
                "decode": "0.6.14",
            },
            "component_builds": {
                "flashinfer-jit-cache": {"installed": False},
            },
            "architecture": {
                "arch": "cuda",
                "device_name": gate.GPU_NAME,
                "sm": 120,
                "kernel_tier": "fa2",
            },
        }
    )
    return {
        "runtime_id": runtime_id,
        "nonce": "a" * 32 if runtime_id == "full" else "b" * 32,
        "engine": copy.deepcopy(
            gate.FULL_RUNTIME_CONFIG if runtime_id == "full" else gate.CHUNK_RUNTIME_CONFIG
        ),
        "topology": {
            "parallelism": {"tensor": 8, "expert": 1, "pipeline": 1},
            "control_backend": "gloo",
            "model_backend": "nccl",
            "rank_gpu_uuid_order": [_uuid(rank) for rank in range(gate.TENSOR_PARALLEL_SIZE)],
            "sampling_ownership": [
                {
                    "rank": rank,
                    "world_size": gate.TENSOR_PARALLEL_SIZE,
                    "sampling_owner": rank == 0,
                }
                for rank in range(gate.TENSOR_PARALLEL_SIZE)
            ],
        },
        "attention": {
            "requested_backend": "flashinfer",
            "selected_backend": "flashinfer",
            "execution_identity": attention_identity,
            "fallback_reason": None,
        },
    }


_STRUCTURE_PROFILES = {
    "batch32-cold": (32, 1, 1, 0, 1, 0, 31, [32], True),
    "pressure": (6, 6, 0, 6, 6, 0, 186, [1, 32], True),
    "alone-cold": (1, 1, 0, 1, 1, 0, 31, [1, 32], False),
    "alone-warm": (1, 1, 0, 1, 0, 1, 31, [1, 32], False),
    "chunked-alone-cold": (5, 5, 0, 5, 4, 1, 31, [1], True),
}


def _structure(scenario: str, request_ids: list[str]) -> dict[str, object]:
    (
        rows,
        calls,
        batched,
        sequential,
        prefill_backend_calls,
        decode_stock_calls,
        graph_executions,
        buckets,
        captures,
    ) = _STRUCTURE_PROFILES[scenario]
    prefill_stats = {
        "enabled": True,
        "capability_gap": None,
        "rows": rows,
        "model_calls": calls,
        "batched_groups": batched,
        "sequential_rows": sequential,
        "backend": [
            {
                "type": "FlashInferBackend",
                "plans": prefill_backend_calls,
                "runs": prefill_backend_calls * 64,
            }
        ],
    }
    decode_stats = {
        "enabled": True,
        "capability_gap": None,
        "requests": 0,
        "positions": 0,
        "model_calls": 0,
        "batched_groups": 0,
        "sequential_positions": 0,
        "graph_groups": 0,
        "graph": {
            "graph_executions": graph_executions,
            "eager_fallbacks": 0,
            "captured_buckets": buckets,
        },
        "backend": [
            {
                "type": "FlashInferBackend",
                "plans": graph_executions
                + decode_stock_calls
                + (3 if captures else 0),
                "runs": (256 if captures else 0) + decode_stock_calls * 64,
                "fast_replay_plans": graph_executions,
                "stock_replay_fallbacks": 0,
                "replay_fallback_reason": None,
            }
        ],
    }
    return {
        "scheduler_steps": gate._expected_scheduler(scenario, request_ids),
        "prefill_rank_stats": [
            {
                "rank": rank,
                "world_size": gate.TENSOR_PARALLEL_SIZE,
                "device": f"cuda:{rank}",
                "stats": copy.deepcopy(prefill_stats),
            }
            for rank in range(gate.TENSOR_PARALLEL_SIZE)
        ],
        "decode_rank_stats": [
            {
                "rank": rank,
                "world_size": gate.TENSOR_PARALLEL_SIZE,
                "device": f"cuda:{rank}",
                "stats": copy.deepcopy(decode_stats),
            }
            for rank in range(gate.TENSOR_PARALLEL_SIZE)
        ],
    }


def _response(
    prompt: dict[str, object],
    *,
    cached_tokens: int,
    answer_offset: int,
) -> dict[str, object]:
    token_ids = [answer_offset + index for index in range(gate.OUTPUT_TOKENS)]
    token_pieces = [f"Ġraw-{token_id}" for token_id in token_ids]
    # Deliberately not ''.join(token_pieces): raw Qwen BPE vocabulary pieces
    # and native detokenized text are independent retained surfaces.
    text = "decoded answer " + " ".join(str(token_id) for token_id in token_ids)
    return {
        "status": "success",
        "token_ids": token_ids,
        "token_pieces": token_pieces,
        "text": text,
        "finish_reason": "length",
        "usage": {
            "prompt_tokens": prompt["token_count"],
            "completion_tokens": gate.OUTPUT_TOKENS,
            "cached_tokens": cached_tokens,
        },
    }


def _request(
    *,
    scenario: str,
    runtime_id: str,
    request_id: str,
    role: str,
    cohort_index: int,
    cohort_size: int,
    prompt: dict[str, object],
    cached_tokens: int,
    answer_offset: int,
    structure: dict[str, object] | None,
    structure_sha256: str,
) -> dict[str, object]:
    return {
        "row_type": "request",
        "row_index": -1,
        "run_id": "cpu-fixture-360",
        "scenario": scenario,
        "runtime_id": runtime_id,
        "runtime_nonce": "a" * 32 if runtime_id == "full" else "b" * 32,
        "request_id": request_id,
        "role": role,
        "cohort_index": cohort_index,
        "cohort_size": cohort_size,
        "prompt": copy.deepcopy(prompt),
        "sampling": copy.deepcopy(gate.SAMPLING_CONFIG),
        "cache": {
            "probe_before": cached_tokens,
            "native_cached_tokens": cached_tokens,
        },
        "response": _response(
            prompt,
            cached_tokens=cached_tokens,
            answer_offset=answer_offset,
        ),
        "error_type": None,
        "retry_count": 0,
        "structure": copy.deepcopy(structure),
        "structure_sha256": structure_sha256,
    }


def _pressure_request(index: int, prompt: dict[str, object]) -> dict[str, object]:
    return {
        "request_id": f"pressure-{index:02d}",
        "prompt": copy.deepcopy(prompt),
        "sampling": copy.deepcopy(gate.SAMPLING_CONFIG),
        "cache": {"probe_before": 0, "native_cached_tokens": 0},
        "response": _response(
            prompt,
            cached_tokens=0,
            answer_offset=10_000 + index * 100,
        ),
        "error_type": None,
        "retry_count": 0,
    }


def _rows() -> list[dict[str, object]]:
    prompts = gate.fixed_prompts()
    run_id = "cpu-fixture-360"
    rows: list[dict[str, object]] = [
        {
            "row_type": "run_start",
            "row_index": -1,
            "schema_version": gate.SCHEMA_VERSION,
            "gate": gate.GATE,
            "run_id": run_id,
            "config": gate.frozen_config(),
            "prompts": copy.deepcopy(prompts),
            "source": _source(),
            "checkpoint": _checkpoint(),
            "hardware": _hardware(),
            "runtimes": {"full": _runtime("full"), "chunk": _runtime("chunk")},
        }
    ]

    batch_ids = ["batch32-target"] + [
        f"batch32-distractor-{index:02d}" for index in range(gate.DISTRACTOR_COUNT)
    ]
    batch_structure = _structure("batch32-cold", batch_ids)
    batch_sha = gate.sha256_json(batch_structure)
    batch_prompts = [prompts["target"], *prompts["distractors"]]
    for index, (request_id, prompt) in enumerate(zip(batch_ids, batch_prompts, strict=True)):
        rows.append(
            _request(
                scenario="batch32-cold",
                runtime_id="full",
                request_id=request_id,
                role="target" if index == 0 else "distractor",
                cohort_index=index,
                cohort_size=gate.BATCH_SIZE,
                prompt=prompt,
                cached_tokens=0,
                answer_offset=100 if index == 0 else 1_000 + index * 100,
                structure=batch_structure if index == 0 else None,
                structure_sha256=batch_sha,
            )
        )

    pressure_ids = [f"pressure-{index:02d}" for index in range(gate.MAX_PRESSURE_REQUESTS)]
    pressure_structure = _structure("pressure", pressure_ids)
    rows.append(
        {
            "row_type": "pressure_summary",
            "row_index": -1,
            "run_id": run_id,
            "scenario": "pressure",
            "runtime_id": "full",
            "runtime_nonce": "a" * 32,
            "target_cache_before": gate.WARM_CACHED_TOKENS,
            "target_cache_after": 0,
            "requests": [
                _pressure_request(index, prompt) for index, prompt in enumerate(prompts["pressure"])
            ],
            "structure": pressure_structure,
            "structure_sha256": gate.sha256_json(pressure_structure),
        }
    )

    for scenario, runtime_id, cached_tokens in (
        ("alone-cold", "full", 0),
        ("alone-warm", "full", gate.WARM_CACHED_TOKENS),
        ("chunked-alone-cold", "chunk", 0),
    ):
        request_id = f"{scenario}-target"
        structure = _structure(scenario, [request_id])
        rows.append(
            _request(
                scenario=scenario,
                runtime_id=runtime_id,
                request_id=request_id,
                role="target",
                cohort_index=0,
                cohort_size=1,
                prompt=prompts["target"],
                cached_tokens=cached_tokens,
                answer_offset=100,
                structure=structure,
                structure_sha256=gate.sha256_json(structure),
            )
        )
    rows.append(
        {
            "row_type": "run_end",
            "row_index": -1,
            "run_id": run_id,
            "status": "complete",
            "errors": [],
            "runtime_count": 2,
            "scenario_count": 5,
        }
    )
    for row_index, row in enumerate(rows):
        row["row_index"] = row_index
    return rows


def _row(rows: list[dict[str, object]], row_type: str, scenario: str | None = None):
    return next(
        row
        for row in rows
        if row["row_type"] == row_type and (scenario is None or row.get("scenario") == scenario)
    )


def _manifest(rows: list[dict[str, object]]) -> dict[str, object]:
    return gate.recompute_manifest(rows, raw_sha256="c" * 64)


def test_public_contract_freezes_prompt_geometry_and_runtime_configs() -> None:
    from kairyu.engine.core.graph_buckets import decode_buckets

    assert gate.SCHEMA_VERSION == "kairyu.a12.batch-invariance.v1"
    assert gate.GATE == "A12-BATCH-INVARIANCE"
    assert len(gate.TARGET_PROMPT_TOKEN_IDS) == gate.TARGET_PROMPT_TOKENS == 129
    assert gate.sha256_text(gate.TARGET_PROMPT_TEXT) == gate.TARGET_PROMPT_TEXT_SHA256
    assert (
        gate.sha256_json(list(gate.TARGET_PROMPT_TOKEN_IDS)) == gate.TARGET_PROMPT_TOKEN_IDS_SHA256
    )
    assert len(gate.DISTRACTOR_TOKEN_IDS) == 31
    assert len(set(gate.DISTRACTOR_TOKEN_IDS)) == 31
    assert len(gate.PRESSURE_PROMPT_TOKEN_IDS) == gate.MAX_PRESSURE_REQUESTS == 6
    assert all(len(ids) == 129 for ids in gate.PRESSURE_PROMPT_TOKEN_IDS)
    assert gate.FULL_RUNTIME_CONFIG["max_num_batched_tokens"] == 512
    assert gate.CHUNK_RUNTIME_CONFIG["max_num_batched_tokens"] == 32
    assert gate.FULL_RUNTIME_CONFIG["num_pages"] == 128
    assert gate.WARM_CACHED_TOKENS == 128
    assert gate.CUDA_GRAPH_BUCKETS == decode_buckets(gate.BATCH_SIZE) == (
        1,
        2,
        4,
        8,
        16,
        24,
        32,
    )
    assert len(gate.REQUIRED_SOURCE_PATHS) == len(set(gate.REQUIRED_SOURCE_PATHS))


def test_valid_fixture_recomputes_a_passing_manifest() -> None:
    manifest = _manifest(_rows())

    assert manifest["passed"] is True
    assert all(manifest["checks"].values())
    assert manifest["checks"]["four_target_native_answers_are_exactly_identical"]
    assert manifest["checks"]["chunked_prefill_is_exactly_32_32_32_32_1"]
    assert manifest["answer_equality"]["exact_native_answer_count"] == 1
    assert manifest["raw"]["row_count"] == 38
    gate.assert_gate(manifest)


def test_operator_assembly_matches_the_pure_38_row_schema_exactly() -> None:
    from bench import batch_invariance_bench as operator

    expected = _rows()
    header = expected[0]

    def observation(row: dict[str, object]) -> dict[str, object]:
        return {
            field: copy.deepcopy(row[field])
            for field in (
                "scenario",
                "runtime_id",
                "runtime_nonce",
                "request_id",
                "role",
                "cohort_index",
                "cohort_size",
                "prompt",
                "sampling",
                "cache",
                "response",
                "error_type",
                "retry_count",
            )
        }

    batch = [row for row in expected if row.get("scenario") == "batch32-cold"]
    pressure = _row(expected, "pressure_summary", "pressure")
    alone_cold = _row(expected, "request", "alone-cold")
    alone_warm = _row(expected, "request", "alone-warm")
    chunked = _row(expected, "request", "chunked-alone-cold")
    assembled = operator._assemble_rows(
        run_id=header["run_id"],
        prompts=header["prompts"],
        source_start=header["source"]["measurement_start"],
        source_end=header["source"]["measurement_end"],
        checkpoint_start=header["checkpoint"]["measurement_start"],
        checkpoint_end=header["checkpoint"]["measurement_end"],
        hardware=header["hardware"],
        runtimes=header["runtimes"],
        full={
            "batch": [observation(row) for row in batch],
            "batch_structure": batch[0]["structure"],
            "pressure": copy.deepcopy(pressure),
            "alone_cold": observation(alone_cold),
            "cold_structure": alone_cold["structure"],
            "alone_warm": observation(alone_warm),
            "warm_structure": alone_warm["structure"],
        },
        chunk={
            "request": observation(chunked),
            "structure": chunked["structure"],
        },
    )

    assert assembled == expected
    assert gate.recompute_manifest(assembled, raw_sha256="d" * 64)["passed"] is True


def test_valid_fixture_binds_fresh_and_reused_graph_capture_counters() -> None:
    rows = _rows()
    for scenario, prefill_plans, prefill_runs, plans, runs, fast in (
        ("batch32-cold", 1, 64, 34, 256, 31),
        ("pressure", 6, 384, 189, 256, 186),
        ("alone-cold", 1, 64, 31, 0, 31),
        ("alone-warm", 0, 0, 32, 64, 31),
        ("chunked-alone-cold", 4, 256, 35, 320, 31),
    ):
        row_type = "pressure_summary" if scenario == "pressure" else "request"
        structure = _row(rows, row_type, scenario)["structure"]
        for rank in structure["prefill_rank_stats"]:
            backend = rank["stats"]["backend"][0]
            assert (backend["plans"], backend["runs"]) == (
                prefill_plans,
                prefill_runs,
            )
        for rank in structure["decode_rank_stats"]:
            backend = rank["stats"]["backend"][0]
            assert (backend["plans"], backend["runs"], backend["fast_replay_plans"]) == (
                plans,
                runs,
                fast,
            )
    assert _manifest(rows)["passed"] is True


def test_canonical_round_trip_replays_raw_and_never_trusts_stored_passed(
    tmp_path: Path,
) -> None:
    raw = tmp_path / gate.RAW_NAME
    retained = tmp_path / gate.MANIFEST_NAME
    written = gate.write_artifact(raw, retained, _rows(), assert_gate=True)

    assert gate.read_jsonl(raw) == _rows()
    assert gate.replay_raw(raw, assert_gate=True) == written
    assert gate.verify_artifact(tmp_path, assert_gate=True) == written
    assert retained.read_text(encoding="utf-8") == gate.canonical_json(written) + "\n"


@pytest.mark.parametrize("surface", ["token_ids", "token_pieces", "text"])
def test_each_native_answer_surface_must_match_all_four_scenarios(surface: str) -> None:
    rows = _rows()
    response = _row(rows, "request", "alone-warm")["response"]
    if surface == "token_ids":
        response[surface][0] += 1
    elif surface == "token_pieces":
        response[surface][0] += "-changed"
    else:
        response[surface] += " changed"

    manifest = _manifest(rows)
    assert manifest["checks"]["four_target_native_answers_are_exactly_identical"] is False
    assert manifest["passed"] is False


def test_raw_bpe_pieces_are_not_required_to_join_to_detokenized_text() -> None:
    rows = _rows()
    target = _row(rows, "request", "batch32-cold")["response"]
    assert "".join(target["token_pieces"]) != target["text"]
    assert _manifest(rows)["passed"] is True


def test_warm_cache_is_page_aligned_128_not_full_prompt_129() -> None:
    rows = _rows()
    warm = _row(rows, "request", "alone-warm")
    warm["cache"] = {"probe_before": 129, "native_cached_tokens": 129}
    warm["response"]["usage"]["cached_tokens"] = 129

    manifest = _manifest(rows)
    assert manifest["checks"]["alone-warm_cache_probe_and_native_usage_exact"] is False
    assert manifest["passed"] is False


def test_six_fixed_pressure_requests_are_required_to_prove_eviction() -> None:
    rows = _rows()
    pressure = _row(rows, "pressure_summary", "pressure")
    pressure["requests"].pop()

    with pytest.raises(gate.EvidenceError, match="pressure request count"):
        _manifest(rows)


def test_pressure_without_observed_target_eviction_fails() -> None:
    rows = _rows()
    pressure = _row(rows, "pressure_summary", "pressure")
    pressure["target_cache_after"] = gate.WARM_CACHED_TOKENS

    manifest = _manifest(rows)
    assert manifest["checks"]["pressure_proves_target_evicted_before_alone_cold"] is False
    assert manifest["passed"] is False


def test_chunked_prefill_shape_tamper_fails_even_when_answer_matches() -> None:
    rows = _rows()
    chunk = _row(rows, "request", "chunked-alone-cold")
    chunk["structure"]["scheduler_steps"][0]["num_tokens"] = [31]
    chunk["structure_sha256"] = gate.sha256_json(chunk["structure"])

    manifest = _manifest(rows)
    assert manifest["checks"]["chunked-alone-cold_scheduler_shape_exact"] is False
    assert manifest["checks"]["four_target_native_answers_are_exactly_identical"] is True
    assert manifest["passed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("graph_executions", 30), ("eager_fallbacks", 1)],
)
def test_all_rank_graph_dispatch_tamper_fails(field: str, value: int) -> None:
    rows = _rows()
    batch = _row(rows, "request", "batch32-cold")
    batch["structure"]["decode_rank_stats"][3]["stats"]["graph"][field] = value
    batch["structure_sha256"] = gate.sha256_json(batch["structure"])
    for distractor in [
        row
        for row in rows
        if row.get("scenario") == "batch32-cold" and row.get("role") == "distractor"
    ]:
        distractor["structure_sha256"] = batch["structure_sha256"]

    manifest = _manifest(rows)
    assert manifest["checks"]["batch32-cold_all_rank_flashinfer_graph_stats_exact"] is False
    assert manifest["passed"] is False


def test_speculative_verification_counters_cannot_contaminate_ordinary_proof() -> None:
    rows = _rows()
    cold = _row(rows, "request", "alone-cold")
    cold["structure"]["decode_rank_stats"][0]["stats"]["requests"] = 1
    cold["structure_sha256"] = gate.sha256_json(cold["structure"])

    manifest = _manifest(rows)
    assert manifest["checks"]["alone-cold_all_rank_flashinfer_graph_stats_exact"] is False
    assert manifest["passed"] is False


def test_missing_rank_and_wrong_graph_bucket_fail() -> None:
    rows = _rows()
    chunk = _row(rows, "request", "chunked-alone-cold")
    chunk["structure"]["prefill_rank_stats"].pop()
    chunk["structure"]["decode_rank_stats"][0]["stats"]["graph"]["captured_buckets"] = [32]
    chunk["structure_sha256"] = gate.sha256_json(chunk["structure"])

    manifest = _manifest(rows)
    assert manifest["checks"]["chunked-alone-cold_all_rank_flashinfer_graph_stats_exact"] is False
    assert manifest["passed"] is False


def test_all_ranks_must_report_the_same_cumulative_graph_bucket_set() -> None:
    rows = _rows()
    cold = _row(rows, "request", "alone-cold")
    cold["structure"]["decode_rank_stats"][4]["stats"]["graph"]["captured_buckets"] = [1, 2, 32]
    cold["structure_sha256"] = gate.sha256_json(cold["structure"])

    manifest = _manifest(rows)
    assert manifest["checks"]["alone-cold_all_rank_flashinfer_graph_stats_exact"] is False
    assert manifest["passed"] is False


def test_prefill_flashinfer_plan_and_per_layer_run_counts_are_exact() -> None:
    rows = _rows()
    batch = _row(rows, "request", "batch32-cold")
    backend = batch["structure"]["prefill_rank_stats"][0]["stats"]["backend"][0]
    assert backend == {"type": "FlashInferBackend", "plans": 1, "runs": 64}
    backend["plans"] = 64
    batch["structure_sha256"] = gate.sha256_json(batch["structure"])
    for distractor in [
        row
        for row in rows
        if row.get("scenario") == "batch32-cold" and row.get("role") == "distractor"
    ]:
        distractor["structure_sha256"] = batch["structure_sha256"]

    assert _manifest(rows)["passed"] is False


def test_source_checkpoint_and_hardware_tamper_are_rejected() -> None:
    rows = _rows()
    rows[0]["source"]["measurement_end"]["git_head"] = "2" * 40
    with pytest.raises(gate.EvidenceError, match="source changed"):
        _manifest(rows)

    rows = _rows()
    rows[0]["checkpoint"]["measurement_end"]["weights"] = rows[0]["checkpoint"]["measurement_end"][
        "weights"
    ][:-1]
    with pytest.raises(gate.EvidenceError, match="17 shards"):
        _manifest(rows)

    rows = _rows()
    rows[0]["hardware"]["gpus"][7]["name"] = "not-the-pinned-gpu"
    with pytest.raises(gate.EvidenceError, match="model differs"):
        _manifest(rows)


def test_attention_execution_identity_is_canonical_and_cross_bound() -> None:
    rows = _rows()
    rows[0]["runtimes"]["full"]["attention"]["execution_identity"] = "x"
    with pytest.raises(gate.EvidenceError, match="cannot parse full attention"):
        _manifest(rows)

    rows = _rows()
    runtime = rows[0]["runtimes"]["full"]
    identity = json.loads(runtime["attention"]["execution_identity"])
    identity["components"]["decode"] = "torch"
    runtime["attention"]["execution_identity"] = gate.canonical_json(identity)
    with pytest.raises(gate.EvidenceError, match="execution selection differs"):
        _manifest(rows)

    rows = _rows()
    runtime = rows[0]["runtimes"]["full"]
    identity = json.loads(runtime["attention"]["execution_identity"])
    identity["architecture"]["sm"] = 90
    runtime["attention"]["execution_identity"] = gate.canonical_json(identity)
    with pytest.raises(gate.EvidenceError, match="architecture differs"):
        _manifest(rows)


def test_full_and_chunk_attention_build_identities_must_match() -> None:
    rows = _rows()
    for runtime_id in ("full", "chunk"):
        runtime = rows[0]["runtimes"][runtime_id]
        identity = json.loads(runtime["attention"]["execution_identity"])
        identity["component_builds"]["flashinfer-jit-cache"] = {
            "installed": True,
            "version": "0.6.14",
            "record_sha256": "1" * 64,
        }
        runtime["attention"]["execution_identity"] = gate.canonical_json(identity)

    assert _manifest(rows)["passed"] is True

    chunk = rows[0]["runtimes"]["chunk"]
    chunk_identity = json.loads(chunk["attention"]["execution_identity"])
    chunk_identity["component_builds"]["flashinfer-jit-cache"]["record_sha256"] = "2" * 64
    chunk["attention"]["execution_identity"] = gate.canonical_json(chunk_identity)

    with pytest.raises(gate.EvidenceError, match="execution identities differ"):
        _manifest(rows)


def test_bool_is_not_accepted_as_an_integer() -> None:
    rows = _rows()
    _row(rows, "request", "alone-cold")["cache"]["probe_before"] = False

    with pytest.raises(gate.EvidenceError, match="probe is invalid"):
        _manifest(rows)


def test_row_indices_order_and_exact_fields_are_strict() -> None:
    rows = _rows()
    rows[5]["row_index"] = 6
    with pytest.raises(gate.EvidenceError, match="indices"):
        _manifest(rows)

    rows = _rows()
    rows[0]["unexpected"] = True
    with pytest.raises(gate.EvidenceError, match="fields differ"):
        _manifest(rows)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{ "a":1}\n',
        b'{"a":1}\r\n',
        b'{"a":1}',
    ],
)
def test_read_jsonl_rejects_duplicate_nonfinite_noncanonical_and_bad_framing(
    tmp_path: Path,
    payload: bytes,
) -> None:
    raw = tmp_path / "bad.jsonl"
    raw.write_bytes(payload)
    with pytest.raises(gate.EvidenceError):
        gate.read_jsonl(raw)


def test_forged_retained_passed_cannot_override_raw_replay(tmp_path: Path) -> None:
    raw = tmp_path / gate.RAW_NAME
    retained = tmp_path / gate.MANIFEST_NAME
    manifest = gate.write_artifact(raw, retained, _rows(), assert_gate=True)
    forged = copy.deepcopy(manifest)
    forged["checks"]["four_target_native_answers_are_exactly_identical"] = False
    forged["passed"] = True
    retained.write_text(gate.canonical_json(forged) + "\n", encoding="utf-8")

    with pytest.raises(gate.EvidenceError, match="independent raw replay"):
        gate.verify_artifact(tmp_path, assert_gate=True)


def test_write_is_exclusive_and_refuses_overwrite(tmp_path: Path) -> None:
    raw = tmp_path / gate.RAW_NAME
    retained = tmp_path / gate.MANIFEST_NAME
    gate.write_artifact(raw, retained, _rows())

    with pytest.raises(gate.EvidenceError, match="overwrite"):
        gate.write_artifact(raw, retained, _rows())


def test_failed_run_is_retained_but_assert_gate_rejects_it(tmp_path: Path) -> None:
    rows = _rows()
    rows[-1]["status"] = "failed"
    rows[-1]["errors"] = ["synthetic capture failure"]
    raw = tmp_path / gate.RAW_NAME
    retained = tmp_path / gate.MANIFEST_NAME

    manifest = gate.write_artifact(raw, retained, rows, assert_gate=False)

    assert raw.is_file() and retained.is_file()
    assert manifest["passed"] is False
    with pytest.raises(gate.EvidenceError, match="run_completed"):
        gate.verify_artifact(tmp_path, assert_gate=True)


def test_manifest_json_is_canonical_and_contains_no_stored_raw_verdict() -> None:
    rows = _rows()
    assert "passed" not in rows[-1]
    manifest = _manifest(rows)
    round_tripped = json.loads(gate.canonical_json(manifest))
    assert round_tripped == manifest
