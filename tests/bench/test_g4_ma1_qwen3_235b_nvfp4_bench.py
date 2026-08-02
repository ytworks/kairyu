"""Offline replay and tamper tests for the real-only G4 M-A1 gate."""

from __future__ import annotations

import copy
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bench import g4_ma1_qwen3_235b_nvfp4_bench as gate


def _source() -> dict[str, object]:
    return {
        "commit": "a" * 40,
        "tracked_dirty": False,
        "files": {name: "b" * 64 for name in gate.BOUND_SOURCE_FILES},
    }


def _checkpoint() -> dict[str, object]:
    weights = [
        {"name": name, "size": size, "sha256": digest}
        for name, size, digest in gate.EXPECTED_WEIGHT_FILES
    ]
    tokenizer = [
        {"name": name, "size": size, "sha256": digest}
        for name, size, digest in gate.EXPECTED_TOKENIZER_FILES
    ]
    return {
        "model": gate.MODEL,
        "revision": gate.MODEL_REVISION,
        "config_sha256": gate.EXPECTED_CONFIG_SHA256,
        "index_sha256": gate.EXPECTED_INDEX_SHA256,
        "hf_quant_config_sha256": gate.EXPECTED_HF_QUANT_CONFIG_SHA256,
        "index_total_size": gate.EXPECTED_INDEX_TOTAL_SIZE,
        "tensor_count": gate.EXPECTED_TENSOR_COUNT,
        "weight_shards": weights,
        "weights_rollup_sha256": gate.sha256_json(weights),
        "tokenizer_files": tokenizer,
        "tokenizer_rollup_sha256": gate.sha256_json(tokenizer),
        "architecture": dict(gate.EXPECTED_ARCHITECTURE),
        "quantization": dict(gate.EXPECTED_QUANTIZATION),
    }


def _environment(arm: str) -> dict[str, object]:
    world_size = gate._arm_world_size(arm)
    packages = (
        {
            "python": "3.12.3",
            "torch": "2.12.1+cu130",
            "tensorrt_llm": "1.2.1",
        }
        if gate._arm_stack(arm) == "reference"
        else {
            "python": "3.12.3",
            "torch": "2.12.1+cu130",
            "kairyu": "0.1.0",
            "flashinfer": "0.6.14",
        }
    )
    gpus = [
        {
            "rank": rank,
            "local_device_index": rank,
            "uuid": f"GPU-{rank:032x}",
            "pci_bus_id": f"00000000:{rank + 1:02X}:00.0",
            "name": gate.EXPECTED_GPU_NAME,
            "sm": gate.EXPECTED_SM,
            "total_memory_bytes": 96 * 1024**3,
            "numa_node": rank // 2,
        }
        for rank in range(world_size)
    ]
    topology_raw = "GPU0 GPU1 CPU Affinity NUMA Affinity\nfixture"
    return {
        "packages": packages,
        "cuda": {"runtime": "13.0", "driver": "590.44", "nccl": "2.29.7"},
        "visibility": {
            "CUDA_VISIBLE_DEVICES": ",".join(str(rank) for rank in range(world_size)),
            "NVIDIA_VISIBLE_DEVICES": ",".join(gpu["uuid"] for gpu in gpus),
            "resolved_rank_uuid_order": [gpu["uuid"] for gpu in gpus],
        },
        "gpus": gpus,
        "topology": {
            "gpu_uuid_order": [gpu["uuid"] for gpu in gpus],
            "gpu_pci_bus_id_order": [gpu["pci_bus_id"] for gpu in gpus],
            "gpu_numa_node_order": [gpu["numa_node"] for gpu in gpus],
            "nvidia_smi_topology_raw": topology_raw,
            "nvidia_smi_topology_sha256": gate.sha256_bytes(
                topology_raw.encode()
            ),
        },
    }


def _runtime(
    arm: str,
    image_repo_digest: str,
    image_config_id: str,
    source: dict[str, object],
) -> dict[str, object]:
    repo_digest = (
        gate.REFERENCE_RUNTIME["image_repo_digest"]
        if gate._arm_stack(arm) == "reference"
        else image_repo_digest
    )
    config_id = (
        gate.REFERENCE_RUNTIME["image_config_id"]
        if gate._arm_stack(arm) == "reference"
        else image_config_id
    )
    observation = {
        "source": "docker-container-inspect",
        "container_id": gate.sha256_json([arm, "container"]),
        "image_repo_digest": repo_digest,
        "image_config_id": config_id,
        "running": True,
        "started_at": "2026-08-01T00:00:00+00:00",
        "inspect_sha256": gate.sha256_json([arm, "inspect"]),
        "source_revision": (
            None if gate._arm_stack(arm) == "reference" else source["commit"]
        ),
    }
    if gate._arm_stack(arm) == "reference":
        return {**gate.REFERENCE_RUNTIME, "container_observation": observation}
    return {
        "name": "kairyu",
        "version": "0.1.0",
        "image_repo_digest": image_repo_digest,
        "image_config_id": image_config_id,
        "backend": "pytorch",
        "result_api": gate.KAIRYU_RESULT_API,
        "source_commit": source["commit"],
        "container_observation": observation,
    }


def _kernels(arm: str) -> list[dict[str, object]]:
    reference = gate._arm_stack(arm) == "reference"
    world_size = gate._arm_world_size(arm)
    experts_per_rank = gate.NUM_EXPERTS // world_size
    projection_count = (
        4 * gate.NUM_LAYERS
        + 3 * gate.NUM_LAYERS * experts_per_rank
    )
    rows = []
    for rank in range(world_size):
        projection_sha = gate.sha256_json([arm, rank])
        execution_evidence = {
            "kind": "singleton-backend-successful-forward",
            "projection_inventory_source": "checkpoint-index",
            "nvfp4_allowed_backends": ["cutlass"],
            "moe_backend": "CUTLASS",
            "successful_forward": True,
        }
        if not reference:
            first_owned = rank * experts_per_rank
            last_owned = first_owned + experts_per_rank - 1
            execution_evidence = {
                "kind": "all-rank-runtime-module-probe",
                "projection_inventory_source": (
                    "rank-local-model.named_modules"
                ),
                "probe": "DistEPModelRunner.ep_kernel_inventory_metadata",
                "successful_forward": True,
                "observed": {
                    "rank": rank,
                    "projection_count": projection_count,
                    "projection_inventory_sha256": projection_sha,
                    "kernel_counts": {
                        "nvfp4:flashinfer_nvfp4": projection_count
                    },
                    "meta_count": {
                        "parameters": 0,
                        "buffers": 0,
                        "total": 0,
                    },
                    "load_info": {
                        "ep_rank": rank,
                        "ep_size": world_size,
                        "owned_expert_count": experts_per_rank,
                        "owned_expert_indices_sha256": gate.sha256_json(
                            list(range(first_owned, last_owned + 1))
                        ),
                        "first_owned_expert": first_owned,
                        "last_owned_expert": last_owned,
                        "quantization_source": "hf_quant_config.json",
                        "quantization_method": "nvfp4",
                        "kv_cache_quant_algo": "FP8",
                        "producer_name": "modelopt",
                        "producer_version": "0.33.0",
                        "checkpoint_tensor_count": (
                            gate.EXPECTED_TENSOR_COUNT
                        ),
                        "rank_loaded_tensor_count": (
                            1977
                            + 12 * gate.NUM_LAYERS * experts_per_rank
                        ),
                        "auxiliary_kv_scale_count": 188,
                    },
                },
            }
        rows.append({
            "rank": rank,
            "sm": gate.EXPECTED_SM,
            "requested": "nvfp4",
            "resolved": (
                "tensorrt_llm.pytorch.MoeConfig.CUTLASS"
                if reference
                else "flashinfer.mm_fp4"
            ),
            "provider": "cutlass" if reference else "flashinfer",
            "provider_version": "3.8" if reference else "0.6.14",
            "checkpoint_layout": "modelopt-nvfp4-group16",
            "weight_dtype": "uint8-packed-e2m1",
            "scale_dtype": "float8_e4m3fn",
            "global_scale_dtype": "float32",
            "activation_dtype": "nvfp4-e2m1",
            "compute_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "quantized_projection_count": projection_count,
            "native_nvfp4_projection_count": projection_count,
            "fallback_projection_count": 0,
            "projection_inventory_sha256": projection_sha,
            "fallback": None,
            "execution_evidence": execution_evidence,
        })
    return rows


def _checkpoint_view(
    arm: str, checkpoint: dict[str, object]
) -> dict[str, object]:
    if gate._arm_stack(arm) == "kairyu":
        return {
            "kind": "original",
            "model_path": "/models/qwen3-235b-nvfp4",
            "hf_quant_config_sha256": gate.EXPECTED_HF_QUANT_CONFIG_SHA256,
        }
    links = [
        {
            "name": row["name"],
            "symlink_target": f"/models/qwen3-235b-nvfp4/{row['name']}",
            "source_device": 1,
            "source_inode": index + 1,
            "runtime_device": 1,
            "runtime_inode": index + 1,
            "size_bytes": row["size"],
        }
        for index, row in enumerate(checkpoint["weight_shards"])
    ]
    return {
        "kind": "bf16-kv-metadata-overlay",
        "source_model_path": "/models/qwen3-235b-nvfp4",
        "runtime_model_path": "/tmp/qwen3-235b-bf16-kv",
        "source_hf_quant_config_sha256": gate.EXPECTED_HF_QUANT_CONFIG_SHA256,
        "runtime_hf_quant_config_sha256": (
            gate.EXPECTED_TRT_RUNTIME_HF_QUANT_CONFIG_SHA256
        ),
        "removed_json_path": "quantization.kv_cache_quant_algo",
        "removed_value": "FP8",
        "kv_cache_config_dtype_argument": "auto",
        "effective_kv_cache_dtype": "bfloat16",
        "weight_links": links,
    }


def _top64(selected: int, *, alternate: int | None = None) -> list[dict[str, object]]:
    rows = [{"token_id": selected, "logprob": -0.1}]
    if alternate is not None:
        rows.append({"token_id": alternate, "logprob": -0.15})
    token = 10_000
    while len(rows) < gate.TOP_LOGPROBS:
        if token not in {selected, alternate}:
            rows.append(
                {"token_id": token, "logprob": -1.0 - 0.01 * len(rows)}
            )
        token += 1
    return rows


def _update_counts(rows: list[dict[str, object]]) -> None:
    end = next(row for row in rows if row["type"] == "run_end")
    end["row_counts_before_end"] = dict(
        sorted(
            Counter(
                str(row["type"])
                for row in rows
                if row["type"] != "run_end"
            ).items()
        )
    )


@pytest.fixture(scope="module")
def valid_rows() -> list[dict[str, object]]:
    source = _source()
    checkpoint = _checkpoint()
    image_repo_digest = "kairyu-local@sha256:" + "d" * 64
    image_config_id = "sha256:" + "c" * 64
    rows: list[dict[str, object]] = [
        {
            "schema_version": gate.SCHEMA_VERSION,
            "type": "run",
            "created_at": "2026-08-01T00:00:00+00:00",
            "config": gate.formal_config(),
            "image_repo_digest": image_repo_digest,
            "image_config_id": image_config_id,
            "source_start": copy.deepcopy(source),
            "checkpoint_start": copy.deepcopy(checkpoint),
        }
    ]
    prompt_tokens: dict[str, list[int]] = {}
    for index, text in enumerate(gate.PROMPTS):
        prompt_id = f"p{index}"
        tokens = [100 + index, 200 + index]
        prompt_tokens[prompt_id] = tokens
        rows.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "type": "prompt",
                "prompt_id": prompt_id,
                "prompt_index": index,
                "text": text,
                "text_sha256": gate.sha256_bytes(text.encode()),
                "token_ids": tokens,
                "token_ids_sha256": gate.sha256_json(tokens),
                "tokenizer_rollup_sha256": checkpoint[
                    "tokenizer_rollup_sha256"
                ],
            }
        )

    origin = datetime(2026, 8, 1, tzinfo=UTC)
    for sequence_index, arm in enumerate(gate.ARMS):
        started = origin + timedelta(minutes=sequence_index * 2)
        finished = started + timedelta(minutes=1)
        environment = _environment(arm)
        rows.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "type": "session",
                "arm": arm,
                "sequence_index": sequence_index,
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "config": gate._session_config(arm),
                "runtime": _runtime(
                    arm, image_repo_digest, image_config_id, source
                ),
                "environment_start": copy.deepcopy(environment),
                "environment_end": copy.deepcopy(environment),
                "checkpoint_start": copy.deepcopy(checkpoint),
                "checkpoint_end": copy.deepcopy(checkpoint),
                "kernel_ranks": _kernels(arm),
                "sampling_ownership": (
                    []
                    if gate._arm_stack(arm) == "reference"
                    else [
                        {
                            "rank": rank,
                            "control_world_size": gate._arm_world_size(arm),
                            "control_backend": "gloo",
                            "model_world_size": gate._arm_world_size(arm),
                            "model_backend": "nccl",
                            "sampling_owner": rank == 0,
                            "sampler_present": rank == 0,
                            "device": f"cuda:{rank}",
                        }
                        for rank in range(gate._arm_world_size(arm))
                    ]
                ),
                "checkpoint_view": _checkpoint_view(arm, checkpoint),
                "attention_backend": {
                    "requested": (
                        "tensorrt_llm"
                        if gate._arm_stack(arm) == "reference"
                        else "auto"
                    ),
                    "resolved": (
                        "tensorrt_llm"
                        if gate._arm_stack(arm) == "reference"
                        else "flashinfer"
                    ),
                    "components": {
                        "prefill": (
                            "tensorrt_llm"
                            if gate._arm_stack(arm) == "reference"
                            else "flashinfer"
                        ),
                        "decode": (
                            "tensorrt_llm"
                            if gate._arm_stack(arm) == "reference"
                            else "flashinfer"
                        ),
                        "kv_mode": "paged-direct",
                    },
                    "identity": f"{arm}-attention",
                },
            }
        )

    for world_size in gate.WORLD_SIZES:
        arm = gate.KAIRYU_ARM[world_size]
        experts_per_rank = gate.NUM_EXPERTS // world_size
        for rank in range(world_size):
            rows.append(
                {
                    "schema_version": gate.SCHEMA_VERSION,
                    "type": "ownership",
                    "arm": arm,
                    "rank": rank,
                    "world_size": world_size,
                    "layers": gate.NUM_LAYERS,
                    "num_experts": gate.NUM_EXPERTS,
                    "owned_expert_indices": list(
                        range(
                            rank * experts_per_rank,
                            (rank + 1) * experts_per_rank,
                        )
                    ),
                    "owned_expert_tensor_count": (
                        12 * gate.NUM_LAYERS * experts_per_rank
                    ),
                    "owned_expert_tensor_names_sha256": gate.sha256_json(
                        [world_size, rank, "experts"]
                    ),
                    "rank_loaded_tensor_count": (
                        1977 + 12 * gate.NUM_LAYERS * experts_per_rank
                    ),
                    "rank_loaded_bytes": 140_000_000_000 // world_size,
                    "loaded_tensor_names_sha256": gate.sha256_json(
                        [world_size, rank, "loaded"]
                    ),
                    "dense_replication_sha256": gate.sha256_json(["dense"]),
                    "router_replication_sha256": gate.sha256_json(["router"]),
                    "shared_expert_replication_sha256": gate.sha256_json(
                        ["shared"]
                    ),
                    "remote_expert_tensor_reads": 0,
                    "unresolved_meta_tensors": 0,
                    "evidence_source": "checkpoint-index-and-build-contract",
                    "load_contract": {
                        "quantization_source": "hf_quant_config.json",
                        "quantization_method": "nvfp4",
                        "checkpoint_kv_cache_quant_algo": "FP8",
                        "runtime_kv_cache_dtype": "bfloat16",
                        "producer_name": "modelopt",
                        "producer_version": "0.33.0",
                        "checkpoint_tensor_count": gate.EXPECTED_TENSOR_COUNT,
                        "auxiliary_kv_scale_count": 188,
                        "ownership": "contiguous-global-expert-blocks",
                    },
                }
            )

    for arm in gate.ARMS:
        for prompt in range(gate.PROMPT_COUNT):
            canonical = list(
                range(
                    1_000 + prompt * gate.POSITIONS,
                    1_000 + (prompt + 1) * gate.POSITIONS,
                )
            )
            rows.append(
                {
                    "schema_version": gate.SCHEMA_VERSION,
                    "type": "free_run",
                    "arm": arm,
                    "prompt_id": f"p{prompt}",
                    "output_token_ids": canonical,
                    "output_token_ids_sha256": gate.sha256_json(canonical),
                    "selected_logprobs": [-0.1] * gate.POSITIONS,
                    "finish_reason": "length",
                    "source_api": gate._result_api(arm),
                }
            )

    for arm in gate.ARMS:
        for prompt in range(gate.PROMPT_COUNT):
            prompt_id = f"p{prompt}"
            canonical = list(
                range(
                    1_000 + prompt * gate.POSITIONS,
                    1_000 + (prompt + 1) * gate.POSITIONS,
                )
            )
            for position, token in enumerate(canonical):
                prefix = prompt_tokens[prompt_id] + canonical[:position]
                rows.append(
                    {
                        "schema_version": gate.SCHEMA_VERSION,
                        "type": "teacher",
                        "arm": arm,
                        "prompt_id": prompt_id,
                        "position": position,
                        "prefix_token_count": len(prefix),
                        "prefix_token_ids_sha256": gate.sha256_json(prefix),
                        "canonical_token_id": token,
                        "selected_token_id": token,
                        "selected_logprob": -0.1,
                        "top_logprobs": _top64(token),
                        "source_api": gate._result_api(arm),
                    }
                )
    rows.append(
        {
            "schema_version": gate.SCHEMA_VERSION,
            "type": "run_end",
            "source_end": copy.deepcopy(source),
            "checkpoint_end": copy.deepcopy(checkpoint),
            "row_counts_before_end": {},
        }
    )
    _update_counts(rows)
    return rows


def _manifest(rows: list[dict[str, object]]) -> dict[str, object]:
    return gate.recompute_manifest(rows, raw_sha256="f" * 64)


def _row(
    rows: list[dict[str, object]],
    row_type: str,
    **identity: object,
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["type"] == row_type
        and all(row.get(key) == value for key, value in identity.items())
    )


def _make_tie_disagreement(
    rows: list[dict[str, object]],
    *,
    world_size: int,
    prompt: int,
    position: int,
) -> None:
    ref = _row(
        rows,
        "teacher",
        arm=gate.REFERENCE_ARM[world_size],
        prompt_id=f"p{prompt}",
        position=position,
    )
    candidate = _row(
        rows,
        "teacher",
        arm=gate.KAIRYU_ARM[world_size],
        prompt_id=f"p{prompt}",
        position=position,
    )
    reference_token = ref["selected_token_id"]
    candidate_token = 20_000 + prompt * gate.POSITIONS + position
    ref["top_logprobs"] = _top64(reference_token, alternate=candidate_token)
    candidate["selected_token_id"] = candidate_token
    candidate["selected_logprob"] = -0.1
    candidate["top_logprobs"] = _top64(candidate_token, alternate=reference_token)


def test_valid_raw_passes_and_reports_fixed_threshold(valid_rows):
    result = _manifest(valid_rows)

    assert result["passed"] is True
    assert result["verdict"] == "PASS"
    assert result["metrics"]["binding_teacher_forced"]["ep2"]["agreed"] == 1024
    assert gate.AGREEMENT_COUNT_MIN == 1014
    assert all(result["checks"].values())


def test_prompt_specific_canonical_association_swap_fails(valid_rows) -> None:
    rows = copy.deepcopy(valid_rows)
    for row in rows:
        if row["type"] != "teacher" or row["arm"] != "reference_ep2":
            continue
        if row["prompt_id"] == "p2":
            row["prompt_id"] = "p10"
        elif row["prompt_id"] == "p10":
            row["prompt_id"] = "p2"

    result = _manifest(rows)
    assert result["passed"] is False
    assert result["checks"]["world_matched_teacher_prefixes_exact"] is False


def test_free_run_association_is_diagnostic_only(valid_rows) -> None:
    rows = copy.deepcopy(valid_rows)
    left = _row(rows, "free_run", arm="reference_ep2", prompt_id="p2")
    right = _row(rows, "free_run", arm="reference_ep2", prompt_id="p10")
    left_tokens = left["output_token_ids"]
    right_tokens = right["output_token_ids"]
    left["output_token_ids"] = right_tokens
    right["output_token_ids"] = left_tokens
    left["output_token_ids_sha256"] = gate.sha256_json(right_tokens)
    right["output_token_ids_sha256"] = gate.sha256_json(left_tokens)

    result = _manifest(rows)
    assert result["passed"] is True
    assert result["checks"]["world_matched_teacher_prefixes_exact"] is True


@pytest.mark.parametrize(
    "tamper",
    ["weight_sha256", "weight_size", "tokenizer_sha256", "tokenizer_size"],
)
def test_checkpoint_requires_exact_official_file_identity(tamper: str) -> None:
    checkpoint = _checkpoint()
    if tamper == "weight_sha256":
        checkpoint["weight_shards"][0]["sha256"] = "0" * 64
        checkpoint["weights_rollup_sha256"] = gate.sha256_json(
            checkpoint["weight_shards"]
        )
    elif tamper == "weight_size":
        checkpoint["weight_shards"][0]["size"] += 1
        checkpoint["weights_rollup_sha256"] = gate.sha256_json(
            checkpoint["weight_shards"]
        )
    elif tamper == "tokenizer_sha256":
        checkpoint["tokenizer_files"][0]["sha256"] = "0" * 64
        checkpoint["tokenizer_rollup_sha256"] = gate.sha256_json(
            checkpoint["tokenizer_files"]
        )
    else:
        checkpoint["tokenizer_files"][0]["size"] += 1
        checkpoint["tokenizer_rollup_sha256"] = gate.sha256_json(
            checkpoint["tokenizer_files"]
        )

    assert gate._checkpoint_valid(checkpoint) is False


@pytest.mark.parametrize(
    "value",
    [
        "repo@sha256:garbage",
        "repo@sha256:" + "a" * 64 + "/suffix",
        " repo@sha256:" + "a" * 64,
        "repo@@sha256:" + "a" * 64,
    ],
)
def test_repo_digest_requires_exact_oci_sha256_identity(value: str) -> None:
    assert gate._repo_digest_valid(value) is False


def test_aligned_invalid_kairyu_repo_digest_fails_raw_replay(valid_rows) -> None:
    rows = copy.deepcopy(valid_rows)
    invalid = "kairyu-local@sha256:garbage"
    _row(rows, "run")["image_repo_digest"] = invalid
    for arm in ("kairyu_ep2", "kairyu_ep4"):
        runtime = _row(rows, "session", arm=arm)["runtime"]
        runtime["image_repo_digest"] = invalid
        runtime["container_observation"]["image_repo_digest"] = invalid

    result = _manifest(rows)
    assert result["passed"] is False
    assert result["checks"]["one_run_and_one_run_end"] is False
    assert result["checks"]["complete_session_provenance"] is False


def test_write_verify_and_raw_only_replay(valid_rows, tmp_path: Path):
    written = gate.write_artifact(tmp_path, valid_rows)

    assert written == gate.verify_artifact(tmp_path)
    assert written == gate.replay_artifact(tmp_path)


@pytest.mark.parametrize("tamper", ["schema", "type"])
def test_unknown_schema_or_type_raises(valid_rows, tamper):
    rows = copy.deepcopy(valid_rows)
    if tamper == "schema":
        rows[1]["schema_version"] = "unknown"
    else:
        rows[1]["type"] = "unknown"

    with pytest.raises(ValueError):
        _manifest(rows)


def test_duplicate_and_missing_teacher_rows_fail_closed(valid_rows):
    duplicate = copy.deepcopy(valid_rows)
    duplicate.insert(
        -1,
        copy.deepcopy(
            _row(
                duplicate,
                "teacher",
                arm="kairyu_ep2",
                prompt_id="p0",
                position=0,
            )
        ),
    )
    _update_counts(duplicate)
    missing = copy.deepcopy(valid_rows)
    missing.remove(_row(missing, "teacher", arm="kairyu_ep4", prompt_id="p63", position=15))
    _update_counts(missing)

    assert _manifest(duplicate)["checks"]["complete_finite_token_id_top64_teacher_rows"] is False
    assert _manifest(missing)["checks"]["complete_finite_token_id_top64_teacher_rows"] is False


def test_nonfinite_json_and_duplicate_object_keys_are_rejected(valid_rows):
    rows = copy.deepcopy(valid_rows)
    teacher = _row(rows, "teacher", arm="kairyu_ep2", prompt_id="p0", position=0)
    teacher["selected_logprob"] = float("nan")

    with pytest.raises(ValueError, match="finite canonical JSON"):
        _manifest(rows)
    with pytest.raises(ValueError, match="non-finite JSON"):
        gate.strict_json_loads('{"value":NaN}')
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        gate.strict_json_loads('{"value":1,"value":2}')


@pytest.mark.parametrize(
    ("tamper", "check"),
    [
        ("config", "formal_config_exact"),
        ("reference_image", "complete_session_provenance"),
        ("checkpoint", "full_checkpoint_identity_stable"),
        ("kernel", "complete_session_provenance"),
        ("sampling_backend", "complete_session_provenance"),
        ("ownership", "complete_contiguous_expert_ownership"),
    ],
)
def test_config_provenance_checkpoint_kernel_and_ownership_tamper_fail(
    valid_rows, tamper, check
):
    rows = copy.deepcopy(valid_rows)
    if tamper == "config":
        _row(rows, "run")["config"]["sampling"]["seed"] += 1
    elif tamper == "reference_image":
        _row(rows, "session", arm="reference_ep2")["runtime"]["image_config_id"] = (
            "sha256:" + "d" * 64
        )
    elif tamper == "checkpoint":
        _row(rows, "run")["checkpoint_start"]["index_sha256"] = "d" * 64
    elif tamper == "kernel":
        _row(rows, "session", arm="kairyu_ep4")["kernel_ranks"][0][
            "fallback"
        ] = "torch"
    elif tamper == "sampling_backend":
        _row(rows, "session", arm="kairyu_ep4")["sampling_ownership"][0][
            "model_backend"
        ] = "gloo"
    else:
        _row(rows, "ownership", arm="kairyu_ep4", rank=1)[
            "owned_expert_indices"
        ][0] = 0

    result = _manifest(rows)
    assert result["passed"] is False
    assert result["checks"][check] is False


def test_reference_teacher_rollout_must_be_canonical(valid_rows):
    rows = copy.deepcopy(valid_rows)
    teacher = _row(rows, "teacher", arm="reference_ep4", prompt_id="p0", position=0)
    wrong = 30_000
    teacher["selected_token_id"] = wrong
    teacher["selected_logprob"] = -0.1
    teacher["top_logprobs"] = _top64(wrong)

    result = _manifest(rows)
    assert result["passed"] is False
    assert result["checks"]["reference_teacher_rollout_chain_exact"] is False


def test_ten_reciprocal_ties_meet_exact_99_percent_floor(valid_rows):
    rows = copy.deepcopy(valid_rows)
    for position in range(10):
        _make_tie_disagreement(
            rows, world_size=2, prompt=0, position=position
        )

    result = _manifest(rows)
    score = result["metrics"]["binding_teacher_forced"]["ep2"]
    assert score["agreed"] == 1014
    assert score["substantive_disagreements"] == 0
    assert result["checks"]["ep2_teacher_agreement_at_least_99_percent"] is True
    assert result["passed"] is True


def test_eleven_ties_do_not_lower_fixed_99_percent_floor(valid_rows):
    rows = copy.deepcopy(valid_rows)
    for position in range(11):
        _make_tie_disagreement(
            rows, world_size=4, prompt=0, position=position
        )

    result = _manifest(rows)
    assert result["metrics"]["binding_teacher_forced"]["ep4"]["agreed"] == 1013
    assert result["checks"]["ep4_teacher_agreement_at_least_99_percent"] is False
    assert result["passed"] is False


def test_substantive_tie_classification_uses_declared_reference_gap(valid_rows):
    rows = copy.deepcopy(valid_rows)
    _make_tie_disagreement(rows, world_size=2, prompt=0, position=0)
    reference = _row(
        rows, "teacher", arm="reference_ep2", prompt_id="p0", position=0
    )
    candidate = _row(
        rows, "teacher", arm="kairyu_ep2", prompt_id="p0", position=0
    )
    # Reference gap is 0.10 (a declared near-tie).  The candidate gap is 0.25,
    # but both cross-runtime token deltas remain <= 0.25.  Requiring the
    # candidate to meet the separate 0.125 reference tie bound would add an
    # undeclared, needlessly stricter acceptance criterion.
    reference["top_logprobs"][1]["logprob"] = -0.2
    candidate["top_logprobs"][1]["logprob"] = -0.35

    result = _manifest(rows)
    score = result["metrics"]["binding_teacher_forced"]["ep2"]
    assert score["disagreements"][0]["reference_tie_gap_nats"] == pytest.approx(0.1)
    assert score["disagreements"][0]["candidate_tie_gap_nats"] == pytest.approx(0.25)
    assert score["substantive_disagreements"] == 0
    assert result["passed"] is True


def test_substantive_or_missing_reciprocal_disagreement_fails(valid_rows):
    rows = copy.deepcopy(valid_rows)
    candidate = _row(rows, "teacher", arm="kairyu_ep2", prompt_id="p0", position=0)
    candidate["selected_token_id"] = 40_000
    candidate["selected_logprob"] = -0.1
    candidate["top_logprobs"] = _top64(40_000)

    result = _manifest(rows)
    score = result["metrics"]["binding_teacher_forced"]["ep2"]
    assert score["substantive_disagreements"] == 1
    assert score["missing_reciprocal_top64"]
    assert result["checks"]["ep2_zero_substantive_disagreements"] is False


def test_selected_logprob_delta_above_quarter_nat_fails(valid_rows):
    rows = copy.deepcopy(valid_rows)
    candidate = _row(rows, "teacher", arm="kairyu_ep4", prompt_id="p0", position=0)
    candidate["selected_logprob"] = -0.36
    candidate["top_logprobs"][0]["logprob"] = -0.36
    # Preserve descending top-64 order while keeping the selected token first.
    for index, top in enumerate(candidate["top_logprobs"][1:], start=1):
        top["logprob"] = -1.0 - 0.01 * index

    result = _manifest(rows)
    assert result["checks"]["ep4_selected_logprob_delta_at_most_0_25"] is False
    assert result["passed"] is False


def test_manifest_or_raw_tamper_cannot_verify(valid_rows, tmp_path: Path):
    gate.write_artifact(tmp_path, valid_rows)
    manifest_path = tmp_path / gate.MANIFEST_NAME
    manifest = gate.strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    manifest["passed"] = False
    manifest_path.write_text(gate.canonical_json(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs"):
        gate.verify_artifact(tmp_path)

    gate.write_artifact(tmp_path, valid_rows)
    raw_path = tmp_path / gate.RAW_NAME
    raw = raw_path.read_text(encoding="utf-8")
    raw_path.write_text(raw.replace('"fallback":null', '"fallback":"torch"', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs"):
        gate.verify_artifact(tmp_path)
    assert gate.replay_artifact(tmp_path)["passed"] is False


def test_run_cli_ingests_capture_without_running_a_gpu(valid_rows, tmp_path: Path):
    capture = tmp_path / "capture.jsonl"
    capture.write_text(
        "".join(gate.canonical_json(row) + "\n" for row in valid_rows),
        encoding="utf-8",
    )
    output = tmp_path / "artifact"

    assert gate.main(
        [
            "run",
            "--capture",
            str(capture),
            "--output-dir",
            str(output),
            "--assert-gate",
        ]
    ) == 0
    assert gate.main(
        ["verify", "--artifact", str(output), "--assert-gate"]
    ) == 0
    assert gate.main(
        ["replay", "--artifact", str(output), "--assert-gate"]
    ) == 0
