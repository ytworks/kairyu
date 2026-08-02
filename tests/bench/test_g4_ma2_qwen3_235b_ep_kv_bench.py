"""Synthetic replay/tamper tests for the real-only G4 M-A2 EP4 operator."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import g4_ma2_qwen3_235b_ep_kv_bench as gate
from kairyu.engine.core.sampling_types import SampledToken


class _StablePieceTokenizer:
    """Small tokenizer whose space-prefixed pieces repeat one-for-one."""

    _piece = re.compile(r" piece(\d{3})")
    eos_token_id = None

    def __init__(self, size: int = 128) -> None:
        self._pieces = [f" piece{token_id:03d}" for token_id in range(size)]

    def vocab(self) -> list[str]:
        return list(self._pieces)

    def decode(self, token_ids) -> str:
        return "".join(self._pieces[token_id] for token_id in token_ids)

    def encode(self, text: str) -> tuple[int, ...]:
        token_ids: list[int] = []
        position = 0
        for match in self._piece.finditer(text):
            if match.start() != position:
                raise ValueError("fixture text is not token aligned")
            token_ids.append(int(match.group(1)))
            position = match.end()
        if position != len(text):
            raise ValueError("fixture text has an unmatched suffix")
        return tuple(token_ids)


def _source() -> dict[str, object]:
    return {
        "commit": "a" * 40,
        "tracked_dirty": False,
        "files": {name: "b" * 64 for name in gate.BOUND_SOURCE_FILES},
    }


def _checkpoint() -> dict[str, object]:
    weights = [
        {"name": name, "size": size, "sha256": digest}
        for name, size, digest in gate.ma1.EXPECTED_WEIGHT_FILES
    ]
    tokenizer = [
        {"name": name, "size": size, "sha256": digest}
        for name, size, digest in gate.ma1.EXPECTED_TOKENIZER_FILES
    ]
    return {
        "model": gate.MODEL,
        "revision": gate.MODEL_REVISION,
        "config_sha256": gate.ma1.EXPECTED_CONFIG_SHA256,
        "index_sha256": gate.ma1.EXPECTED_INDEX_SHA256,
        "hf_quant_config_sha256": gate.ma1.EXPECTED_HF_QUANT_CONFIG_SHA256,
        "index_total_size": gate.ma1.EXPECTED_INDEX_TOTAL_SIZE,
        "tensor_count": gate.ma1.EXPECTED_TENSOR_COUNT,
        "weight_shards": weights,
        "weights_rollup_sha256": gate.sha256_json(weights),
        "tokenizer_files": tokenizer,
        "tokenizer_rollup_sha256": gate.sha256_json(tokenizer),
        "architecture": dict(gate.ma1.EXPECTED_ARCHITECTURE),
        "quantization": dict(gate.ma1.EXPECTED_QUANTIZATION),
    }


def _environment() -> dict[str, object]:
    gpus = [
        {
            "rank": rank,
            "local_device_index": rank,
            "uuid": f"GPU-{rank:032x}",
            "pci_bus_id": f"00000000:{rank + 1:02X}:00.0",
            "name": gate.ma1.EXPECTED_GPU_NAME,
            "sm": gate.ma1.EXPECTED_SM,
            "total_memory_bytes": 96 * 1024**3,
            "numa_node": rank // 2,
        }
        for rank in range(gate.EXPERT_PARALLEL_SIZE)
    ]
    topology_raw = "GPU0 GPU1 GPU2 GPU3 CPU Affinity NUMA Affinity\nfixture"
    return {
        "packages": {
            "python": "3.12.3",
            "torch": "2.12.1+cu130",
            "kairyu": "0.1.0",
            "flashinfer": "0.6.14",
        },
        "cuda": {"runtime": "13.0", "driver": "590.44", "nccl": "2.29.7"},
        "visibility": {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
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


def _container(source: dict[str, object]) -> dict[str, object]:
    return {
        "source": "docker-container-inspect",
        "container_id": "1" * 64,
        "image_repo_digest": "fixture/kairyu@sha256:" + "d" * 64,
        "image_config_id": "sha256:" + "c" * 64,
        "running": True,
        "started_at": "2026-08-02T00:00:00+00:00",
        "inspect_sha256": "e" * 64,
        "source_revision": source["commit"],
    }


def _runtime(
    source: dict[str, object], container: dict[str, object]
) -> dict[str, object]:
    return {
        "name": "kairyu",
        "version": "0.1.0",
        "image_repo_digest": container["image_repo_digest"],
        "image_config_id": container["image_config_id"],
        "backend": "pytorch",
        "result_api": gate.ma1.KAIRYU_RESULT_API,
        "source_commit": source["commit"],
        "container_observation": container,
    }


def _attention_backend() -> dict[str, object]:
    return {
        "requested": "auto",
        "resolved": "flashinfer",
        "components": {
            "prefill": "flashinfer",
            "decode": "flashinfer",
            "kv_mode": "paged-direct",
        },
        "identity": gate.canonical_json(
            {
                "requested": "auto",
                "resolved": "flashinfer",
                "kv_mode": "paged-direct",
            }
        ),
    }


def _accounting_start() -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "world_size": gate.EXPERT_PARALLEL_SIZE,
            "capture_action": "start",
            "observation_limit": gate.ACCOUNTING_OBSERVATION_LIMIT,
            "overflow": False,
            "capture_error": None,
            "request_count": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "observations_sha256": gate.sha256_json([]),
            "observations": [],
        }
        for rank in range(gate.EXPERT_PARALLEL_SIZE)
    ]


def _kernel_rows() -> list[dict[str, object]]:
    experts_per_rank = gate.ma1.NUM_EXPERTS // gate.EXPERT_PARALLEL_SIZE
    routed_count = 3 * gate.ma1.NUM_LAYERS * experts_per_rank
    projection_count = 4 * gate.ma1.NUM_LAYERS + routed_count
    dense_count = projection_count - routed_count
    rows = []
    for rank in range(gate.EXPERT_PARALLEL_SIZE):
        first_owned = rank * experts_per_rank
        rows.append(
            {
                "rank": rank,
                "projection_count": projection_count,
                "projection_inventory_sha256": gate.sha256_json(
                    ["projection", rank]
                ),
                "kernel_counts": {
                    "nvfp4:flashinfer_cutlass_fused_moe": routed_count,
                    "nvfp4:flashinfer_nvfp4": dense_count,
                },
                "fused_moe": {
                    "backend": "flashinfer.cutlass_fused_moe",
                    "global_expert_count": gate.ma1.NUM_EXPERTS,
                    "local_expert_count": experts_per_rank,
                    "local_expert_start": first_owned,
                    "weight_order": "up_gate",
                    "scale_layout": "128x4",
                    "output_dtype": "bfloat16",
                    "ep_partial": True,
                    "enable_alltoall": False,
                    "block_count": gate.ma1.NUM_LAYERS,
                    "successful_forward_block_count": gate.ma1.NUM_LAYERS,
                },
                "attention_output": {
                    "backend": "flashinfer.mm_fp4",
                    "placement": "row_parallel",
                    "parallel_size": gate.EXPERT_PARALLEL_SIZE,
                    "rank": rank,
                    "shard_axis": 1,
                    "global_in_features": 8_192,
                    "local_in_features": 8_192 // gate.EXPERT_PARALLEL_SIZE,
                    "out_features": 4_096,
                    "partial_dtype": "bfloat16",
                    "block_count": gate.ma1.NUM_LAYERS,
                    "successful_forward_block_count": gate.ma1.NUM_LAYERS,
                },
                "meta_count": {"parameters": 0, "buffers": 0, "total": 0},
                "load_info": {
                    "ep_rank": rank,
                    "ep_size": gate.EXPERT_PARALLEL_SIZE,
                    "owned_expert_count": experts_per_rank,
                    "owned_expert_indices_sha256": gate.sha256_json(
                        list(
                            range(
                                first_owned,
                                first_owned + experts_per_rank,
                            )
                        )
                    ),
                    "first_owned_expert": first_owned,
                    "last_owned_expert": first_owned + experts_per_rank - 1,
                    "quantization_source": "hf_quant_config.json",
                    "quantization_method": "nvfp4",
                    "kv_cache_quant_algo": "FP8",
                    "producer_name": "modelopt",
                    "producer_version": "0.33.0",
                    "checkpoint_tensor_count": gate.ma1.EXPECTED_TENSOR_COUNT,
                    "rank_loaded_tensor_count": (
                        gate.ma1.LOADED_NON_EXPERT_TENSOR_COUNT
                        + gate.ma1.NUM_LAYERS * experts_per_rank * 12
                    ),
                    "auxiliary_kv_scale_count": 188,
                },
            }
        )
    return rows


def _refresh_accounting(evidence: dict[str, object]) -> None:
    observations = evidence["observations"]
    evidence["prompt_tokens"] = sum(row["prompt_tokens"] for row in observations)
    evidence["cached_tokens"] = sum(row["cached_tokens"] for row in observations)
    evidence["observations_sha256"] = gate.sha256_json(observations)


def _refresh_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
    return gate._with_row_indices(rows)


class _CaptureRunner:
    """CPU runner that records the same first-prefill evidence as EP ranks."""

    def __init__(self) -> None:
        self.observations: list[dict[str, object]] = []
        self.seen: set[str] = set()

    def start_ep_kv_accounting(self):
        return tuple(copy.deepcopy(_accounting_start()))

    def sampling_ownership_metadata(self):
        return tuple(copy.deepcopy(gate._sampling_ownership_expected()))

    def execute(self, scheduled, states):
        sampled = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if chunk.request_id not in self.seen:
                self.seen.add(chunk.request_id)
                self.observations.append(
                    {
                        "sequence": len(self.observations),
                        "request_id": chunk.request_id,
                        "prompt_tokens": len(state.prompt_token_ids),
                        "cached_tokens": state.num_cached_tokens,
                        "prefill_start": (
                            state.computed_prompt - chunk.num_tokens
                        ),
                        "first_num_tokens": chunk.num_tokens,
                        "first_is_prefill": chunk.is_prefill,
                        "prompt_token_ids_sha256": gate.sha256_json(
                            list(state.prompt_token_ids)
                        ),
                        "page_count": len(state.page_ids),
                        "page_ids_sha256": gate.sha256_json(
                            list(state.page_ids)
                        ),
                    }
                )
            if not chunk.is_prefill or state.prefill_done:
                sampled[chunk.request_id] = (SampledToken(1),)
        return sampled

    def finish_ep_kv_accounting(self):
        rows = []
        for rank in range(gate.EXPERT_PARALLEL_SIZE):
            evidence = {
                "rank": rank,
                "world_size": gate.EXPERT_PARALLEL_SIZE,
                "capture_action": "finish",
                "observation_limit": gate.ACCOUNTING_OBSERVATION_LIMIT,
                "overflow": False,
                "capture_error": None,
                "request_count": len(self.observations),
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "observations_sha256": "",
                "observations": copy.deepcopy(self.observations),
            }
            _refresh_accounting(evidence)
            rows.append(evidence)
        return tuple(rows)

    def release(self, _request_id: str) -> None:
        pass


def _capture_launcher_type(mutation: str):
    class _CaptureLauncher:
        def __init__(self, *_args, **_kwargs) -> None:
            self.runner = _CaptureRunner()
            self.attention_backend_identity = "fixture-flashinfer"
            resolved = "torch" if mutation == "attention" else "flashinfer"
            self.attention_backend_decision = SimpleNamespace(
                requested="auto",
                resolved=resolved,
                components={
                    "prefill": resolved,
                    "decode": resolved,
                    "kv_mode": "paged-direct",
                },
            )

        def parallelism_metadata(self):
            return gate._expected_parallelism_metadata()

        def ep_kernel_inventory_metadata(self):
            rows = _kernel_rows()
            if mutation == "kernel":
                rows[2]["fused_moe"]["successful_forward_block_count"] -= 1
            return tuple(rows)

        def shutdown(self) -> None:
            pass

    return _CaptureLauncher


def _patch_capture_host(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source()
    checkpoint = _checkpoint()
    environment = _environment()
    container = _container(source)
    monkeypatch.setattr(
        gate, "_source_snapshot", lambda: copy.deepcopy(source)
    )
    monkeypatch.setattr(
        gate.ma1_capture,
        "_checkpoint_snapshot",
        lambda _path: copy.deepcopy(checkpoint),
    )
    monkeypatch.setattr(
        gate.ma1_capture,
        "_environment",
        lambda _arm: copy.deepcopy(environment),
    )
    monkeypatch.setattr(
        gate.ma1_capture,
        "_container_observation",
        lambda *_args, **_kwargs: copy.deepcopy(container),
    )
    monkeypatch.setattr(
        gate.ma1_capture, "_package_version", lambda _name: "0.1.0"
    )


@pytest.fixture(scope="module")
def passing_rows() -> list[dict[str, object]]:
    source = _source()
    checkpoint = _checkpoint()
    environment = _environment()
    container = _container(source)
    prompts = gate.build_workload(_StablePieceTokenizer())
    trace = gate.trace_descriptor(
        prompts,
        tokenizer_sha256=checkpoint["tokenizer_rollup_sha256"],
    )
    run = {
        "schema_version": gate.SCHEMA_VERSION,
        "gate": gate.GATE,
        "type": "run",
        "created_at": "2026-08-02T00:00:00+00:00",
        "config": gate.formal_config(),
        "trace": trace,
        "runtime": _runtime(source, container),
        "source_start": copy.deepcopy(source),
        "checkpoint_start": copy.deepcopy(checkpoint),
        "checkpoint_view": {
            "kind": "original",
            "model_path": "/models/Qwen3-235B-A22B-NVFP4",
            "hf_quant_config_sha256": checkpoint[
                "hf_quant_config_sha256"
            ],
        },
        "environment_start": copy.deepcopy(environment),
        "topology": gate._topology_value(
            gate._expected_parallelism_metadata(), environment
        ),
        "sampling_ownership": gate._sampling_ownership_expected(),
        "attention_backend": _attention_backend(),
        "accounting_start": _accounting_start(),
    }
    rows: list[dict[str, object]] = [run]
    observations: list[dict[str, object]] = []
    for prompt in prompts:
        request_id = gate._request_id(
            prompt.sequence, prompt.session, prompt.turn
        )
        rows.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "gate": gate.GATE,
                "type": "request",
                "sequence": prompt.sequence,
                "request_id": request_id,
                "session": prompt.session,
                "turn": prompt.turn,
                "prompt_token_ids_sha256": prompt.prompt_token_ids_sha256,
                "expected_cached_tokens": prompt.expected_cached_tokens,
                "status": "success",
                "finish_reason": "length",
                "error_type": None,
                "output_token_ids": [1],
                "usage": {
                    "prompt_tokens": len(prompt.prompt_token_ids),
                    "cached_tokens": prompt.expected_cached_tokens,
                    "completion_tokens": 1,
                },
            }
        )
        hashes = gate._prefix_block_hashes(prompt.prompt_token_ids)
        cached_pages = prompt.expected_cached_tokens // gate.PAGE_SIZE
        rows.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "gate": gate.GATE,
                "type": "cache_event",
                "event_sequence": prompt.sequence,
                "request_id": request_id,
                "payload": {
                    "type": "BlockStored",
                    "block_hashes": list(hashes[cached_pages:]),
                    "parent_block_hash": (
                        hashes[cached_pages - 1] if cached_pages else None
                    ),
                    "token_ids": list(
                        prompt.prompt_token_ids[
                            prompt.expected_cached_tokens :
                        ]
                    ),
                    "block_size": gate.PAGE_SIZE,
                },
            }
        )
        observations.append(
            {
                "sequence": prompt.sequence,
                "request_id": request_id,
                "prompt_tokens": len(prompt.prompt_token_ids),
                "cached_tokens": prompt.expected_cached_tokens,
                "prefill_start": prompt.expected_cached_tokens,
                "first_num_tokens": (
                    len(prompt.prompt_token_ids)
                    - prompt.expected_cached_tokens
                ),
                "first_is_prefill": True,
                "prompt_token_ids_sha256": prompt.prompt_token_ids_sha256,
                "page_count": len(prompt.prompt_token_ids) // gate.PAGE_SIZE,
                "page_ids_sha256": gate.sha256_json(
                    list(range(len(prompt.prompt_token_ids) // gate.PAGE_SIZE))
                ),
            }
        )
    for rank in range(gate.EXPERT_PARALLEL_SIZE):
        evidence = {
            "rank": rank,
            "world_size": gate.EXPERT_PARALLEL_SIZE,
            "capture_action": "finish",
            "observation_limit": gate.ACCOUNTING_OBSERVATION_LIMIT,
            "overflow": False,
            "capture_error": None,
            "request_count": gate.EXPECTED_REQUESTS,
            "prompt_tokens": gate.EXPECTED_PROMPT_TOKENS,
            "cached_tokens": gate.EXPECTED_THEORETICAL_CACHED_TOKENS,
            "observations_sha256": gate.sha256_json(observations),
            "observations": copy.deepcopy(observations),
        }
        rows.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "gate": gate.GATE,
                "type": "rank_accounting",
                "rank": rank,
                "evidence": evidence,
            }
        )
    for rank, evidence in enumerate(_kernel_rows()):
        rows.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "gate": gate.GATE,
                "type": "kernel_rank",
                "rank": rank,
                "evidence": evidence,
            }
        )
    rows.append(
        {
            "schema_version": gate.SCHEMA_VERSION,
            "gate": gate.GATE,
            "type": "run_end",
            "finished_at": "2026-08-02T01:00:00+00:00",
            "source_end": copy.deepcopy(source),
            "checkpoint_end": copy.deepcopy(checkpoint),
            "environment_end": copy.deepcopy(environment),
            "row_counts_before_end": {},
        }
    )
    return _refresh_rows(rows)


def test_fixed_geometry_and_capacity_are_exact() -> None:
    prompts = gate.build_workload(_StablePieceTokenizer())

    assert len(prompts) == gate.EXPECTED_REQUESTS == 512
    assert sum(len(prompt.prompt_token_ids) for prompt in prompts) == 557_056
    assert sum(prompt.expected_cached_tokens for prompt in prompts) == 491_008
    assert 491_008 / 557_056 == pytest.approx(0.8814338235294118)
    assert gate.MAX_RETAINED_PAGES == 4_128
    assert gate.ACTIVE_REQUEST_GUARD_PAGES == 1
    assert gate.NUM_PAGES == 4_129
    assert gate.KV_BYTES_PER_PAGE == 3_080_192
    assert gate.formal_config()["binding"]["rank_rows_are_not_rate_samples"] is True
    assert set(gate.ma1.BOUND_SOURCE_FILES) < set(gate.BOUND_SOURCE_FILES)
    assert {
        "Dockerfile.cuda",
        "kairyu/engine/config_validation.py",
        "kairyu/engine/kairyu_backend.py",
        "kairyu/engine/openai_backend.py",
        "kairyu/entrypoints/server/health.py",
    } <= set(gate.BOUND_SOURCE_FILES)


def test_replay_block_hash_chain_matches_the_production_radix_event() -> None:
    from kairyu.engine.core.radix_kv import RadixKVCache

    events: list[dict[str, object]] = []
    prompt = tuple(range(10 * gate.PAGE_SIZE))
    cache = RadixKVCache(
        num_pages=11,
        page_size=gate.PAGE_SIZE,
        event_sink=events.append,
    )
    allocation = cache.allocate(prompt)
    cache.mark_computed(allocation)

    assert len(events) == 1
    assert events[0]["block_hashes"] == list(gate._prefix_block_hashes(prompt))
    assert events[0]["parent_block_hash"] is None


def test_passing_artifact_verifies_and_raw_only_replays(
    passing_rows: list[dict[str, object]], tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact"
    manifest = gate.write_artifact(artifact, passing_rows)

    assert manifest["passed"] is True
    assert all(manifest["checks"].values())
    assert manifest["metrics"]["request_count"] == 512
    assert manifest["metrics"]["rank_count"] == 4
    assert manifest["metrics"]["logical_prompt_tokens"] == 557_056
    assert manifest["metrics"]["logical_cached_tokens"] == 491_008
    assert manifest["metrics"]["logical_engine_cache_rate"] == pytest.approx(
        0.8814338235294118
    )
    assert manifest["metrics"]["rank_rows_are_not_aggregated_into_rate"] is True
    assert gate.verify_artifact(artifact, assert_gate=True) == manifest

    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["passed"] = False
    (artifact / gate.MANIFEST_NAME).write_text(
        gate.canonical_json(tampered_manifest) + "\n",
        encoding="utf-8",
    )
    assert gate.replay_artifact(artifact, assert_gate=True)["passed"] is True
    with pytest.raises(ValueError, match="independent raw replay"):
        gate.verify_artifact(artifact)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("below_floor", "logical_engine_cache_rate_strictly_above_0_80"),
        ("missing_request", "every_request_present_exactly_once"),
        (
            "rank_cached_drift",
            "every_rank_observation_matches_logical_engine_truth",
        ),
        ("capture_error", "ep4_rank_accounting_complete"),
        ("rank_page_drift", "rank_page_identity_has_no_drift"),
        (
            "event_reordered",
            "raw_radix_blockstored_events_complete_and_exact",
        ),
        (
            "event_removed",
            "raw_radix_blockstored_events_complete_and_exact",
        ),
        (
            "kernel_drift",
            "ep4_topology_sampling_and_kernel_evidence_exact",
        ),
        (
            "source_drift",
            "source_checkpoint_container_and_gpu_start_end_exact",
        ),
        (
            "gpu_drift",
            "source_checkpoint_container_and_gpu_start_end_exact",
        ),
    ],
)
def test_semantic_tampering_replays_to_a_failed_gate(
    passing_rows: list[dict[str, object]],
    tmp_path: Path,
    mutation: str,
    failed_check: str,
) -> None:
    rows = copy.deepcopy(passing_rows)
    if mutation == "below_floor":
        for row in rows:
            if row["type"] == "request":
                row["usage"]["cached_tokens"] = 0
            elif row["type"] == "rank_accounting":
                for observation in row["evidence"]["observations"]:
                    observation["cached_tokens"] = 0
                    observation["prefill_start"] = 0
                    observation["first_num_tokens"] = observation[
                        "prompt_tokens"
                    ]
                _refresh_accounting(row["evidence"])
    elif mutation == "missing_request":
        rows.remove(next(row for row in rows if row["type"] == "request"))
    elif mutation == "rank_cached_drift":
        wrapper = next(
            row
            for row in rows
            if row["type"] == "rank_accounting" and row["rank"] == 3
        )
        observation = wrapper["evidence"]["observations"][0]
        observation["cached_tokens"] = 16
        observation["prefill_start"] = 16
        observation["first_num_tokens"] -= 16
        _refresh_accounting(wrapper["evidence"])
    elif mutation == "rank_page_drift":
        wrapper = next(
            row
            for row in rows
            if row["type"] == "rank_accounting" and row["rank"] == 2
        )
        wrapper["evidence"]["observations"][1]["page_ids_sha256"] = "f" * 64
        _refresh_accounting(wrapper["evidence"])
    elif mutation == "capture_error":
        wrapper = next(
            row
            for row in rows
            if row["type"] == "rank_accounting" and row["rank"] == 1
        )
        wrapper["evidence"]["capture_error"] = "RuntimeError: fixture"
    elif mutation == "event_reordered":
        indices = [
            index for index, row in enumerate(rows) if row["type"] == "cache_event"
        ][:2]
        rows[indices[0]], rows[indices[1]] = rows[indices[1]], rows[indices[0]]
    elif mutation == "event_removed":
        event = next(row for row in rows if row["type"] == "cache_event")
        event["payload"] = {
            "type": "BlockRemoved",
            "block_hashes": event["payload"]["block_hashes"],
        }
    elif mutation == "kernel_drift":
        kernel = next(row for row in rows if row["type"] == "kernel_rank")
        kernel["evidence"]["fused_moe"][
            "successful_forward_block_count"
        ] -= 1
    elif mutation == "source_drift":
        end = next(row for row in rows if row["type"] == "run_end")
        first = gate.BOUND_SOURCE_FILES[0]
        end["source_end"]["files"][first] = "c" * 64
    else:
        end = next(row for row in rows if row["type"] == "run_end")
        changed = "GPU-" + "f" * 32
        end["environment_end"]["gpus"][0]["uuid"] = changed
        end["environment_end"]["visibility"][
            "resolved_rank_uuid_order"
        ][0] = changed
        end["environment_end"]["topology"]["gpu_uuid_order"][0] = changed

    rows = _refresh_rows(rows)
    artifact = tmp_path / mutation
    manifest = gate.write_artifact(artifact, rows)

    assert manifest["passed"] is False
    assert manifest["checks"][failed_check] is False
    assert gate.replay_artifact(artifact)["passed"] is False
    assert gate.verify_artifact(artifact)["passed"] is False
    with pytest.raises(RuntimeError, match="does not pass"):
        gate.verify_artifact(artifact, assert_gate=True)


def test_raw_byte_tampering_is_rejected_before_semantic_replay(
    passing_rows: list[dict[str, object]], tmp_path: Path
) -> None:
    artifact = tmp_path / "raw-tamper"
    gate.write_artifact(artifact, passing_rows)
    raw = artifact / gate.RAW_NAME
    text = raw.read_text(encoding="utf-8")
    raw.write_text(
        text.replace('"output_token_ids":[1]', '"output_token_ids":[2]', 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw SHA-256"):
        gate.verify_artifact(artifact)


def test_manifest_tampering_is_rejected_with_unchanged_raw(
    passing_rows: list[dict[str, object]], tmp_path: Path
) -> None:
    artifact = tmp_path / "manifest-tamper"
    manifest = gate.write_artifact(artifact, passing_rows)
    manifest["checks"]["all_512_requests_successful"] = False
    (artifact / gate.MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="independent raw replay"):
        gate.verify_artifact(artifact)


def test_run_gate_retains_a_completed_but_rejected_capture(
    passing_rows: list[dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = copy.deepcopy(passing_rows)
    request = next(row for row in rows if row["type"] == "request")
    request["status"] = "failed"
    request["error_type"] = "FixtureError"
    rows = _refresh_rows(rows)
    monkeypatch.setattr(gate, "capture_formal_rows", lambda **_kwargs: rows)

    artifact = tmp_path / "completed-failure"
    manifest = gate.run_gate(
        model_path=Path("/models/fixture"),
        container_inspect=Path("/results/inspect.json"),
        image_repo_digest="fixture/image@sha256:" + "d" * 64,
        image_config_id="sha256:" + "c" * 64,
        output_dir=artifact,
    )

    assert manifest["passed"] is False
    assert (artifact / gate.RAW_NAME).is_file()
    assert (artifact / gate.MANIFEST_NAME).is_file()
    assert gate.replay_artifact(artifact)["passed"] is False


@pytest.mark.parametrize(
    ("mutation", "retained_path"),
    [
        ("kernel", ("evidence", "fused_moe")),
        ("attention", ("attention_backend", "resolved")),
    ],
)
def test_real_capture_flow_retains_completed_binding_mismatches(
    mutation: str,
    retained_path: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_capture_host(monkeypatch)

    rows = gate.capture_formal_rows(
        model_path=Path("/models/fixture"),
        container_inspect=Path("/results/container-inspect.json"),
        image_repo_digest="fixture/kairyu@sha256:" + "d" * 64,
        image_config_id="sha256:" + "c" * 64,
        launcher_type=_capture_launcher_type(mutation),
        tokenizer=_StablePieceTokenizer(),
    )
    artifact = tmp_path / mutation
    manifest = gate.write_artifact(artifact, rows)

    assert manifest["passed"] is False
    assert manifest["checks"][
        "ep4_topology_sampling_and_kernel_evidence_exact"
    ] is False
    assert (artifact / gate.RAW_NAME).is_file()
    assert (artifact / gate.MANIFEST_NAME).is_file()
    run = rows[0]
    if retained_path[0] == "attention_backend":
        assert run[retained_path[0]][retained_path[1]] == "torch"
    else:
        kernel = next(
            row
            for row in rows
            if row["type"] == "kernel_rank" and row["rank"] == 2
        )
        assert kernel[retained_path[0]][retained_path[1]][
            "successful_forward_block_count"
        ] == gate.ma1.NUM_LAYERS - 1
    assert gate.verify_artifact(artifact)["passed"] is False


def test_source_snapshot_rejects_untracked_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_calls: list[tuple[str, ...]] = []

    def fake_git(*args: str) -> str:
        if args[0] == "ls-files":
            return args[-1]
        if args[0] == "rev-parse":
            return "a" * 40
        if args[0] == "status":
            status_calls.append(args)
            return "?? injected.py"
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(gate, "_git", fake_git)

    with pytest.raises(ValueError, match="clean bound source commit"):
        gate._source_snapshot()

    assert status_calls == [
        ("status", "--porcelain", "--untracked-files=all")
    ]
