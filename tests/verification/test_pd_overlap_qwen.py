from __future__ import annotations

import copy
import statistics

import pytest

from verification.l1.performance import pd_overlap_qwen as gate


def _checkpoint() -> dict:
    weights = [
        {
            "file": "model-00001-of-00001.safetensors",
            "bytes": 1,
            "sha256": "a" * 64,
        }
    ]
    snapshot = {
        "architecture": {
            "model_type": "qwen3",
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
        },
        "weights": weights,
        "weights_sha256": gate._checkpoint_weights_sha256(weights),
        "metadata_sha256": {"config.json": "b" * 64},
        "tokenizer_sha256": {"tokenizer.json": "c" * 64},
    }
    return {
        **snapshot,
        "measurement_end": copy.deepcopy(snapshot),
        "stable_across_measurement": True,
    }


def _hardware() -> dict:
    return {
        "arch": "cuda",
        "profile_device_count": 2,
        "cuda_available": True,
        "visible_device_count": 2,
        "peer_access_matrix": [[None, True], [True, None]],
        "devices": [
            {
                "index": index,
                "name": "test-gpu",
                "capability": [8, 0],
                "uuid": f"GPU-{index}",
                "pci": f"0000:0{index}:00",
                "total_memory_bytes": 80 * 1024**3,
            }
            for index in range(2)
        ],
        "runtime": {
            "torch_cuda": "12.8",
            "device_capabilities": [[8, 0], [8, 0]],
            "nvidia_smi": ["gpu0", "gpu1"],
            "nvidia_smi_topology": ["GPU0 GPU1", "GPU1 X"],
        },
    }


def _p2p_probe() -> dict:
    selected = {"prefill": 0, "decode": 1}
    digest = {"sha256": "d" * 64, "bytes": gate.P2P_PROBE_RAW_BYTES}
    publication = {
        "type": "BlockStored",
        "block_hashes": gate._expected_probe_block_hashes(),
        "parent_block_hash": None,
        "token_ids": list(
            range(
                1,
                gate.P2P_PROBE_NUM_PAGES * gate.P2P_PROBE_PAGE_SIZE + 1,
            )
        ),
        "block_size": gate.P2P_PROBE_PAGE_SIZE,
    }

    def topology() -> dict:
        return {
            "source_copy_stream": {
                "device": "cuda:0",
                "cuda_stream": 11,
            },
            "source_compute_stream": {
                "device": "cuda:0",
                "cuda_stream": 0,
            },
            "destination_dependency_stream": {
                "device": "cuda:1",
                "cuda_stream": 12,
            },
            "destination_compute_stream": {
                "device": "cuda:1",
                "cuda_stream": 0,
            },
            "gate_devices": ["cuda:0", "cuda:1"],
            "source_copy_stream_is_compute_stream": False,
            "destination_dependency_stream_is_compute_stream": False,
            "copy_streams_are_distinct": True,
        }

    blocking = {
        "condition": "blocking",
        "status": "complete",
        "topology": topology(),
        "completion_token_created": False,
        "completion_ready_at_return": True,
        "pending_completions_at_return": 0,
        "pending_completions_final": 0,
        "settled_completion_outcomes_final": 0,
        "publication_events_at_return": [copy.deepcopy(publication)],
        "publication_events_final": [copy.deepcopy(publication)],
        "source_copy_interval_ms": [0.0, 1.0],
        "destination_dependency_interval_ms": [0.0, 1.0],
        "source_role_work_interval_ms": [2.0, 3.0],
        "destination_role_work_interval_ms": [2.0, 3.0],
        "source_role_overlap": False,
        "destination_role_overlap": False,
        "destination_kv": copy.deepcopy(digest),
    }
    deferred = {
        "condition": "deferred",
        "status": "complete",
        "topology": topology(),
        "completion_token_created": True,
        "completion_ready_at_return": False,
        "pending_completions_at_return": 1,
        "pending_completions_final": 0,
        "settled_completion_outcomes_final": 0,
        "publication_events_at_return": [],
        "publication_events_final": [copy.deepcopy(publication)],
        "source_copy_interval_ms": [0.0, 3.0],
        "destination_dependency_interval_ms": [0.0, 3.0],
        "source_role_work_interval_ms": [1.0, 2.0],
        "destination_role_work_interval_ms": [1.0, 2.0],
        "source_role_overlap": True,
        "destination_role_overlap": True,
        "destination_kv": copy.deepcopy(digest),
    }
    tokens = tuple(
        range(
            1,
            gate.P2P_PROBE_NUM_PAGES * gate.P2P_PROBE_PAGE_SIZE + 1,
        )
    )
    return {
        "measurement_kind": "real-qwen3-32b-layout-p2p-event-probe",
        "selected_devices": selected,
        "peer_access": {
            "prefill_to_decode": True,
            "decode_to_prefill": True,
        },
        "geometry": {
            "num_layers": gate.P2P_PROBE_NUM_LAYERS,
            "num_pages": gate.P2P_PROBE_NUM_PAGES,
            "page_size": gate.P2P_PROBE_PAGE_SIZE,
            "num_kv_heads": gate.P2P_PROBE_NUM_KV_HEADS,
            "head_dim": gate.P2P_PROBE_HEAD_DIM,
            "dtype": "torch.bfloat16",
            "logical_tokens": len(tokens),
            "raw_bytes": gate.P2P_PROBE_RAW_BYTES,
            "digest_order": "K then V, allocation page order",
            "token_ids_sha256": gate._canonical_json_sha256(tokens),
        },
        "event_policy": (
            "binding source-copy and destination-dependency CUDA-event "
            "interval intersection; no elapsed-time or throughput threshold"
        ),
        "source_kv_before": copy.deepcopy(digest),
        "source_kv_after": copy.deepcopy(digest),
        "conditions": {
            "blocking": blocking,
            "deferred": deferred,
        },
    }


def _request_evidence() -> list[dict]:
    rows = []
    for index, expected in enumerate(gate._expected_request_config()):
        token_ids = [100 + index, 200 + index]
        rows.append(
            {
                **expected,
                "prompt_token_ids": token_ids,
                "prompt_tokens": len(token_ids),
                "text_sha256": gate._sha256_text(expected["text"]),
                "prompt_token_ids_sha256": (
                    gate._canonical_json_sha256(token_ids)
                ),
            }
        )
    return rows


def _condition_run(
    condition: str,
    requests: list[dict],
) -> dict:
    batches = []
    condition_scale = 2.0 if condition == "deferred" else 1.0
    for round_index in range(gate.DEFAULT_ROUNDS):
        evidence = [
            row for row in requests if row["round"] == round_index
        ]
        wall_s = condition_scale * (4.0 + round_index)
        rows = []
        for index, request in enumerate(evidence):
            token_ids = [
                1_000 + 100 * round_index + 10 * index + position
                for position in range(gate.MAX_NEW_TOKENS)
            ]
            text = f"completion-{round_index}-{request['case_id']}"
            rows.append(
                {
                    "request_id": request["request_id"],
                    "status": "complete",
                    "finished": True,
                    "stream_updates": gate.MAX_NEW_TOKENS,
                    "start_offset_s": 0.01 * index,
                    "ttft_s": 0.10 + 0.01 * index,
                    "completion_latency_s": 1.0 + 0.1 * index,
                    "token_ids": token_ids,
                    "text": text,
                    "text_sha256": gate._sha256_text(text),
                    "finish_reason": "length",
                    "usage": {
                        "prompt_tokens": request["prompt_tokens"],
                        "completion_tokens": len(token_ids),
                        "cached_tokens": 0,
                    },
                }
            )
        completion_tokens = sum(
            row["usage"]["completion_tokens"] for row in rows
        )
        batches.append(
            {
                "condition": condition,
                "round": round_index,
                "status": "complete",
                "request_ids": [
                    request["request_id"] for request in evidence
                ],
                "wall_s": wall_s,
                "completed_requests_per_s": len(rows) / wall_s,
                "output_tokens_per_s": completion_tokens / wall_s,
                "total_completion_tokens": completion_tokens,
                "median_ttft_s": statistics.median(
                    row["ttft_s"] for row in rows
                ),
                "median_completion_latency_s": statistics.median(
                    row["completion_latency_s"] for row in rows
                ),
                "requests": rows,
            }
        )
    return {
        "condition": condition,
        "measurement_kind": gate.MEASUREMENT_KIND,
        "status": "complete",
        "load_s_excluded_from_measurement": 1.0,
        "topology": gate._expected_topology(
            condition, {"prefill": 0, "decode": 1}
        ),
        "warmup": {
            "status": "complete",
            "finished": True,
            "token_ids": [1, 2],
            "wall_s": 0.5,
        },
        "rounds": batches,
    }


@pytest.fixture
def valid_artifact(monkeypatch: pytest.MonkeyPatch) -> dict:
    requests = _request_evidence()
    prompt_digest = gate._prompt_token_ids_sha256(requests)
    monkeypatch.setattr(
        gate, "EXPECTED_PROMPT_TOKEN_IDS_SHA256", prompt_digest
    )
    monkeypatch.setattr(
        gate, "_source_is_clean_and_bound", lambda _source: True
    )
    artifact = {
        "schema_version": gate.SCHEMA_VERSION,
        "measurement_kind": gate.MEASUREMENT_KIND,
        "source": {},
        "hardware": _hardware(),
        "checkpoint": _checkpoint(),
        "p2p_probe": _p2p_probe(),
        "config": {
            "conditions": list(gate.CONDITIONS),
            "condition_execution_order": list(gate.CONDITIONS),
            "selected_devices": {"prefill": 0, "decode": 1},
            "minimum_visible_cuda_devices": gate.MINIMUM_VISIBLE_GPUS,
            "rounds": gate.DEFAULT_ROUNDS,
            "num_pages": gate.DEFAULT_NUM_PAGES,
            "page_size": gate.DEFAULT_PAGE_SIZE,
            "max_num_batched_tokens": gate.MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": gate.MAX_NUM_SEQS,
            "max_new_tokens": gate.MAX_NEW_TOKENS,
            "warmup_new_tokens": gate.WARMUP_NEW_TOKENS,
            "pd_separation": True,
            "tensor_parallel_size": 1,
            "decode_mode": "eager",
            "latency_clock": "time.perf_counter_ns",
            "cuda_boundary_synchronization": True,
            "p2p_probe": gate._expected_p2p_probe_config(),
            "performance_gate": False,
            "performance_policy": gate.PERFORMANCE_POLICY,
            "prompt_token_ids_sha256": prompt_digest,
            "measurement_source": (
                "production KairyuBackend with real checkpoint"
            ),
        },
        "requests": requests,
        "runs": {
            condition: _condition_run(condition, requests)
            for condition in gate.CONDITIONS
        },
    }
    checks, summary, diagnostics = gate.verify_evidence(artifact)
    artifact["checks"] = checks
    artifact["summary"] = summary
    artifact["diagnostics"] = diagnostics
    artifact["gate_passed"] = all(checks.values())
    return artifact


def test_valid_artifact_replays_without_gpu(valid_artifact: dict) -> None:
    checks, summary, diagnostics = gate.verify_evidence(
        valid_artifact, verify_stored=True
    )

    assert all(checks.values())
    assert checks["stored_replay_matches"]
    assert summary["exact_output_parity"]
    assert summary["p2p_event_overlap_and_raw_kv_parity"]
    assert all(row["binding"] is False for row in diagnostics)


def test_slower_deferred_measurement_is_diagnostic_not_a_failure(
    valid_artifact: dict,
) -> None:
    checks, _summary, diagnostics = gate.verify_evidence(valid_artifact)

    assert all(checks.values())
    aggregate = diagnostics[-1]["deferred_over_blocking"]
    assert aggregate["wall"] > 1.0
    assert aggregate["output_tokens_per_s"] < 1.0


def test_output_token_mismatch_fails_correctness_gate(
    valid_artifact: dict,
) -> None:
    artifact = copy.deepcopy(valid_artifact)
    artifact["runs"]["deferred"]["rounds"][0]["requests"][0][
        "token_ids"
    ][0] += 1

    checks, summary, _diagnostics = gate.verify_evidence(artifact)

    assert checks["blocking_production_run_complete"]
    assert checks["deferred_production_run_complete"]
    assert not checks["exact_blocking_deferred_output_parity"]
    assert not summary["exact_output_parity"]


def test_synthetic_or_skipped_run_cannot_pass(valid_artifact: dict) -> None:
    synthetic = copy.deepcopy(valid_artifact)
    synthetic["measurement_kind"] = "synthetic"
    synthetic["runs"]["deferred"]["measurement_kind"] = "synthetic"
    synthetic_checks, _summary, _diagnostics = gate.verify_evidence(synthetic)

    skipped = copy.deepcopy(valid_artifact)
    skipped["runs"]["deferred"]["skipped"] = True
    skipped_checks, _summary, _diagnostics = gate.verify_evidence(skipped)

    assert not synthetic_checks["fixed_real_measurement_configuration"]
    assert not synthetic_checks["deferred_production_run_complete"]
    assert not skipped_checks["deferred_production_run_complete"]


def test_mutated_stored_verdict_fails_offline_replay(
    valid_artifact: dict,
) -> None:
    artifact = copy.deepcopy(valid_artifact)
    artifact["summary"]["outputs_compared"] += 1

    checks, _summary, _diagnostics = gate.verify_evidence(
        artifact, verify_stored=True
    )

    assert not checks["stored_replay_matches"]


def test_p2p_timeline_overlap_is_recomputed(
    valid_artifact: dict,
) -> None:
    artifact = copy.deepcopy(valid_artifact)
    deferred = artifact["p2p_probe"]["conditions"]["deferred"]
    deferred["source_copy_interval_ms"] = [0.0, 0.5]

    checks, summary, _diagnostics = gate.verify_evidence(artifact)

    assert not checks["real_p2p_event_overlap_and_raw_kv_parity"]
    assert not summary["p2p_event_overlap_and_raw_kv_parity"]


def test_p2p_raw_kv_digest_mismatch_fails(
    valid_artifact: dict,
) -> None:
    artifact = copy.deepcopy(valid_artifact)
    artifact["p2p_probe"]["conditions"]["deferred"]["destination_kv"][
        "sha256"
    ] = "f" * 64

    checks, _summary, _diagnostics = gate.verify_evidence(artifact)

    assert not checks["real_p2p_event_overlap_and_raw_kv_parity"]


def test_p2p_requires_dedicated_source_and_destination_streams(
    valid_artifact: dict,
) -> None:
    artifact = copy.deepcopy(valid_artifact)
    topology = artifact["p2p_probe"]["conditions"]["deferred"]["topology"]
    topology["source_copy_stream"] = copy.deepcopy(
        topology["source_compute_stream"]
    )

    checks, _summary, _diagnostics = gate.verify_evidence(artifact)

    assert not checks["real_p2p_event_overlap_and_raw_kv_parity"]


def test_checkpoint_must_be_identical_after_measurement(
    valid_artifact: dict,
) -> None:
    artifact = copy.deepcopy(valid_artifact)
    artifact["checkpoint"]["measurement_end"]["metadata_sha256"][
        "config.json"
    ] = "f" * 64

    checks, _summary, _diagnostics = gate.verify_evidence(artifact)

    assert not checks["full_qwen3_32b_checkpoint_provenance"]


def test_selected_pair_must_support_peer_access(
    valid_artifact: dict,
) -> None:
    artifact = copy.deepcopy(valid_artifact)
    artifact["hardware"]["peer_access_matrix"][0][1] = False

    checks, _summary, _diagnostics = gate.verify_evidence(artifact)

    assert not checks["real_distinct_cuda_device_pair"]


def test_malformed_selected_pair_fails_closed(
    valid_artifact: dict,
) -> None:
    artifact = copy.deepcopy(valid_artifact)
    artifact["config"]["selected_devices"] = {"prefill": 0}

    checks, _summary, _diagnostics = gate.verify_evidence(artifact)

    assert not checks["fixed_real_measurement_configuration"]
    assert not checks["real_distinct_cuda_device_pair"]
    assert not checks["real_p2p_event_overlap_and_raw_kv_parity"]
    assert not checks["blocking_production_run_complete"]
    assert not checks["deferred_production_run_complete"]


def test_source_binding_covers_all_kairyu_python_and_dependencies() -> None:
    expected_python = {
        path.relative_to(gate._REPO_ROOT).as_posix()
        for path in (gate._REPO_ROOT / "kairyu").rglob("*.py")
    }

    assert expected_python <= set(gate.SOURCE_BOUND_PATHS)
    assert {
        gate.SOURCE_PATH,
        "verification/l1/correctness/parity_tp.py",
        "pyproject.toml",
        "uv.lock",
    } <= set(gate.SOURCE_BOUND_PATHS)


def test_probe_block_hashes_match_radix_publication() -> None:
    from kairyu.engine.core.radix_kv import RadixKVCache

    events = []
    cache = RadixKVCache(
        gate.P2P_PROBE_NUM_PAGES,
        gate.P2P_PROBE_PAGE_SIZE,
        event_sink=events.append,
    )
    tokens = tuple(
        range(
            1,
            gate.P2P_PROBE_NUM_PAGES * gate.P2P_PROBE_PAGE_SIZE + 1,
        )
    )
    allocation = cache.allocate(tokens)
    cache.mark_computed(allocation)

    assert events[0]["block_hashes"] == gate._expected_probe_block_hashes()
