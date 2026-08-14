"""Focused CPU contracts for the replayable G4 M-A3 operator."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from verification.l1.performance import g2_a6_vllm_bench as a6
from verification.l1.performance import g4_ma3_sglang_bench as gate


def _token_row(sequence: int, source_index: int, *, graph: bool = False) -> dict[str, object]:
    token_ids = [10_000 + sequence] * 64 if graph else [sequence + 1] * 4
    text = (f" graph-{sequence}" if graph else f" prompt-{source_index}") or " prompt"
    return {
        "sequence": sequence,
        "prompt_text": text,
        "prompt_text_sha256": gate.sha256_text(text),
        "prompt_token_ids": token_ids,
        "prompt_tokens": len(token_ids),
        "prompt_token_ids_sha256": gate.sha256_json(token_ids),
        **({} if graph else {"source_index": source_index}),
    }


def _trace_bundle(path: Path) -> tuple[dict[str, object], str]:
    dataset_sha256 = a6.SHAREGPT_DATASET_SHA256
    trace = gate.hashed_descriptor(
        {
            "schema_version": 1,
            "workload": "sharegpt",
            "tokenizer_sha256": gate.TOKENIZER_SHA256,
            "dataset": {
                "format": "ShareGPT conversations JSON",
                "repository": a6.SHAREGPT_DATASET_REPO,
                "revision": a6.SHAREGPT_DATASET_REVISION,
                "file": a6.SHAREGPT_DATASET_FILE,
                "sha256": dataset_sha256,
                "selection_seed": a6.SHAREGPT_DATASET_SEED,
                "candidate_conversation_count": 1_000,
                "minimum_prompt_tokens": a6.SHAREGPT_MIN_PROMPT_TOKENS,
                "maximum_prompt_tokens": a6.SHAREGPT_MAX_PROMPT_TOKENS,
            },
            "warmup": [
                _token_row(sequence, gate.SHAREGPT_REQUESTS + sequence)
                for sequence in range(gate.SERIAL_WARMUP_REQUESTS)
            ],
            "requests": [
                _token_row(sequence, sequence) for sequence in range(gate.SHAREGPT_REQUESTS)
            ],
        }
    )
    graph = gate.hashed_descriptor(
        {
            "schema_version": 1,
            "construction": "arm-neutral-stable-token-repeat-v1",
            "tokenizer_sha256": gate.TOKENIZER_SHA256,
            "prompts": [
                _token_row(sequence, sequence, graph=True)
                for sequence in range(gate.GRAPH_WARMUP_REQUESTS)
            ],
            "batch_sizes": list(gate.GRAPH_WARMUP_BATCH_SIZES),
            "request_count": gate.GRAPH_WARMUP_REQUESTS,
            "completion_tokens": gate.GRAPH_WARMUP_OUTPUT_TOKENS,
        }
    )
    bundle = {
        "schema_version": gate.TRACE_SCHEMA_VERSION,
        "benchmark": gate.benchmark_config(dataset_sha256),
        "trace": trace,
        "graph_warmup": graph,
    }
    path.write_text(gate.canonical_json(bundle) + "\n", encoding="utf-8")
    assert gate.load_trace_bundle(path, dataset_sha256=dataset_sha256) == bundle
    return bundle, gate.sha256_file(path)


def _runtime(arm: str, candidate: str) -> dict[str, object]:
    depth = int(candidate.removeprefix("depth-")) if arm == "kairyu" else None
    return {
        "served_model_name": gate.SERVED_MODEL_NAME,
        "world_size": gate.GPU_COUNT,
        "tensor_parallel_size": 1 if arm == "kairyu" else gate.GPU_COUNT,
        "data_parallel_size": gate.GPU_COUNT,
        "expert_parallel_size": gate.GPU_COUNT,
        "attention_data_parallel": True,
        "request_owned_attention": True,
        "kv_cache_owner": "request-owner",
        "sampling_owner": "request-owner",
        "moe_a2a_backend": (
            gate.KAIRYU_MOE_A2A_BACKEND
            if arm == "kairyu"
            else gate.DEFAULT_SGLANG_MOE_A2A_BACKEND
        ),
        "moe_runner_backend": (
            gate.KAIRYU_MOE_RUNNER_BACKEND
            if arm == "kairyu"
            else gate.SGLANG_MOE_RUNNER_BACKEND
        ),
        "pipeline_depth": depth,
        "overlap_mode": candidate if arm == "sglang" else None,
        "prefill_cuda_graph": "disabled",
        "decode_cuda_graph": True,
        "cuda_graph_max_batch": gate.KAIRYU_CUDA_GRAPH_MAX_BATCH if arm == "kairyu" else None,
        "cuda_graph_max_pages": gate.KAIRYU_CUDA_GRAPH_MAX_PAGES if arm == "kairyu" else None,
        "cuda_graph_warmup_iters": (
            gate.KAIRYU_CUDA_GRAPH_WARMUP_ITERS if arm == "kairyu" else None
        ),
        "cuda_graph_buckets": list(gate.KAIRYU_CUDA_GRAPH_BUCKETS) if arm == "kairyu" else None,
        "max_running_requests": gate.MAX_RUNNING_REQUESTS,
        "configured_chunked_prefill_tokens": gate.CONFIGURED_PREFILL_TOKENS,
        "resolved_chunked_prefill_tokens_per_owner": (gate.RESOLVED_PREFILL_TOKENS_PER_OWNER),
        "configured_max_prefill_tokens": (
            gate.CONFIGURED_PREFILL_TOKENS
            if arm == "kairyu"
            else gate.SGLANG_MAX_PREFILL_TOKENS_PER_OWNER
        ),
        "resolved_max_prefill_tokens_per_owner": gate.RESOLVED_PREFILL_TOKENS_PER_OWNER,
        "page_size": gate.PAGE_SIZE,
        "max_total_tokens": gate.MAX_TOTAL_TOKENS,
        "max_total_tokens_per_owner": gate.MAX_TOTAL_TOKENS_PER_OWNER,
        "kv_cache_dtype": gate.KV_CACHE_DTYPE,
        "prefix_cache": True,
        "scheduler_policy": "fcfs",
        "speculative_decoding": False,
        "access_log": False,
        "resolved_argv": (
            [
                "python",
                "verification/l1/performance/g4_ma3_kairyu_server.py",
                "--pipeline-depth",
                str(depth),
                "--host",
                "0.0.0.0",
                "--port",
                "30000",
            ]
            if arm == "kairyu"
            else gate.expected_sglang_argv(overlap_mode=candidate)
        ),
    }


def _graph_witness(*, replays: int) -> dict[str, object]:
    return {
        "model": gate.SERVED_MODEL_NAME,
        "engine_backend": "kairyu",
        "parallelism": "expert_parallel",
        "expert_parallel_size": gate.GPU_COUNT,
        "attention_placement": "request_owned_data_parallel",
        "attention_output_placement": "replicated",
        "attention_output_parallel_size": 1,
        "attention_output_partial_dtype": None,
        "execution_mode": "request-owned-attention-dp",
        "pipeline_depth": 5,
        "decode_mode": "cuda_graph",
        "kv_cache_dtype": gate.KV_CACHE_DTYPE,
        "attention_data_parallel_size": gate.GPU_COUNT,
        "attention_tensor_parallel_size": 1,
        "moe_dispatcher": "nvfp4_allgather_reduce_scatter",
        "sampling_ownership": "request_owner",
        "kv_cache_ownership": "request_owner",
        "cuda_graph_decode": True,
        "cuda_graph_buckets": list(gate.KAIRYU_CUDA_GRAPH_BUCKETS),
        "cuda_graph_captures": len(gate.KAIRYU_CUDA_GRAPH_BUCKETS),
        "cuda_graph_replays": replays,
        "cuda_graph_eager_fallbacks": 0,
        "moe_collective_transport": {
            "selected_backend": "direct_nccl_ctypes",
            "fallback_backend": "torch.distributed:nccl",
            "direct_nccl_active": True,
            "direct_nccl_library": "libnccl.so.2",
            "direct_nccl_version": "2.29.7",
            "selection_reason": "direct NCCL runtime is available",
        },
    }


class _BackendsResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _BackendsClient:
    def __init__(self, response: _BackendsResponse) -> None:
        self.response = response
        self.urls: list[str] = []

    async def get(self, url: str) -> _BackendsResponse:
        self.urls.append(url)
        return self.response


def _checkpoint_capture(
    phase: str,
    *,
    started_ns: int = 1,
    completed_ns: int = 2,
) -> dict[str, object]:
    return gate.hashed_descriptor(
        {
            "schema_version": gate.CHECKPOINT_CAPTURE_SCHEMA_VERSION,
            "type": "checkpoint_capture",
            "phase": phase,
            "capture_started_ns": started_ns,
            "capture_completed_ns": completed_ns,
            "capture_container": {
                "id": ("a" if phase == "start" else "b") * 64,
                "image_repo_digest": f"ghcr.io/ytworks/kairyu@sha256:{'c' * 64}",
                "image_config_id": f"sha256:{'d' * 64}",
                "source_mount": {
                    "source": "/tmp/g4-ma3-source",
                    "destination": gate.CAPTURE_SOURCE_ROOT,
                    "read_only": True,
                },
                "model_mount": {
                    "source": (
                        "volume:kairyu-qwen3-235b-nvfp4/"
                        f"{gate.MODEL_VOLUME_SUBPATH}"
                    ),
                    "destination": gate.CHECKPOINT_ROOT,
                    "read_only": True,
                },
                "network_mode": "none",
                "gpu_device_requests": [],
                "resolved_argv": ["sleep", "infinity"],
            },
            "source": {
                "repository": "https://github.com/ytworks/kairyu",
                "commit": "e" * 40,
                "clean": True,
            },
            "checkpoint_volume": {
                "name": "kairyu-qwen3-235b-nvfp4",
                "driver": "local",
                "scope": "local",
                "mountpoint": "/var/lib/docker/volumes/kairyu-qwen3-235b-nvfp4/_data",
                "subpath": gate.MODEL_VOLUME_SUBPATH,
                "rw_consumer_count": 0,
            },
            "checkpoint": gate.hashed_descriptor(gate.expected_checkpoint()),
        }
    )


def _runtime_witness(arm: str, runtime: dict[str, object]) -> dict[str, object]:
    if arm == "kairyu":
        return {
            "endpoint": "/backends",
            "response": {
                "role": "engine-host",
                "engines": [
                    {
                        "model": gate.SERVED_MODEL_NAME,
                        "engine_backend": "kairyu",
                        "parallelism": "expert_parallel",
                        "expert_parallel_size": gate.GPU_COUNT,
                        "attention_data_parallel_size": gate.GPU_COUNT,
                        "attention_tensor_parallel_size": 1,
                        "attention_placement": "request_owned_data_parallel",
                        "attention_output_placement": "replicated",
                        "execution_mode": "request-owned-attention-dp",
                        "pipeline_depth": runtime["pipeline_depth"],
                        "decode_mode": "cuda_graph",
                        "kv_cache_dtype": gate.KV_CACHE_DTYPE,
                        "moe_dispatcher": gate.KAIRYU_MOE_RUNNER_BACKEND,
                        "sampling_ownership": "request_owner",
                        "kv_cache_ownership": "request_owner",
                        "cuda_graph_decode": True,
                        "cuda_graph_buckets": list(gate.KAIRYU_CUDA_GRAPH_BUCKETS),
                        "cuda_graph_captures": 0,
                        "cuda_graph_replays": 0,
                        "cuda_graph_eager_fallbacks": 0,
                        "moe_collective_transport": {
                            "selected_backend": gate.KAIRYU_MOE_A2A_BACKEND,
                            "direct_nccl_active": True,
                        },
                    }
                ],
            },
        }
    resolved = {
        "served_model_name": gate.SERVED_MODEL_NAME,
        "tp_size": gate.GPU_COUNT,
        "dp_size": gate.GPU_COUNT,
        "ep_size": gate.GPU_COUNT,
        "enable_dp_attention": True,
        "load_balance_method": "round_robin",
        "dtype": "bfloat16",
        "kv_cache_dtype": gate.KV_CACHE_DTYPE,
        "fp4_gemm_runner_backend": gate.SGLANG_FP4_GEMM_BACKEND,
        "moe_a2a_backend": gate.DEFAULT_SGLANG_MOE_A2A_BACKEND,
        "moe_runner_backend": gate.SGLANG_MOE_RUNNER_BACKEND,
        "page_size": gate.PAGE_SIZE,
        "max_running_requests": gate.MAX_RUNNING_REQUESTS,
        "max_total_tokens": gate.MAX_TOTAL_TOKENS_PER_OWNER,
        "chunked_prefill_size": gate.RESOLVED_PREFILL_TOKENS_PER_OWNER,
        "max_prefill_tokens": gate.SGLANG_MAX_PREFILL_TOKENS_PER_OWNER,
        "schedule_policy": "fcfs",
        "disable_radix_cache": False,
        "speculative_algorithm": None,
        "log_requests": False,
        "log_level_http": "warning",
        "enable_single_batch_overlap": False,
        "enable_two_batch_overlap": False,
    }
    return {
        "endpoint": "/server_info",
        "response": {
            "version": gate.SGLANG_VERSION,
            "model_path": gate.SERVED_MODEL_NAME,
            **resolved,
            "max_total_num_tokens": gate.MAX_TOTAL_TOKENS_PER_OWNER,
            "cuda_graph_config": {
                "decode": {
                    "backend": "full",
                    "max_bs": gate.KAIRYU_CUDA_GRAPH_MAX_BATCH,
                },
                "prefill": {"backend": "disabled"},
            },
            "internal_states": [dict(resolved) for _ in range(gate.GPU_COUNT)],
        },
    }


def _provenance(
    *, arm: str, candidate: str, ordinal: int, server_started_ns: int
) -> dict[str, object]:
    sglang = arm == "sglang"
    environment = {
        "host_id": "fixture-host",
        "driver_version": "590.0",
        "cuda_version": gate.SGLANG_CUDA_VERSION if sglang else "13.0",
        "nccl_version": gate.SGLANG_NCCL_VERSION if sglang else "2.29.7",
        "torch_version": gate.SGLANG_TORCH_VERSION if sglang else "2.11.0+cu130",
        "flashinfer_version": (gate.SGLANG_FLASHINFER_VERSION if sglang else "0.6.14"),
        "engine_name": "sglang" if sglang else "kairyu",
        "engine_version": gate.SGLANG_VERSION if sglang else "0.1.0",
        "gpu_jobs_exclusive": True,
    }
    source = (
        {
            "repository": gate.SGLANG_SOURCE_REPOSITORY,
            "version": gate.SGLANG_VERSION,
            "tag": gate.SGLANG_TAG,
            "tag_object": gate.SGLANG_TAG_OBJECT,
            "commit": gate.SGLANG_SOURCE_COMMIT,
            "clean": True,
        }
        if sglang
        else {
            "repository": "https://github.com/ytworks/kairyu",
            "commit": "e" * 40,
            "clean": True,
        }
    )
    container = {
        "id": f"{ordinal + 1:064x}",
        "image_repo_digest": (
            gate.SGLANG_IMAGE_REPO_DIGEST if sglang else f"ghcr.io/ytworks/kairyu@sha256:{'a' * 64}"
        ),
        "image_platform_digest": (
            gate.SGLANG_AMD64_MANIFEST_DIGEST if sglang else f"sha256:{'b' * 64}"
        ),
        "image_config_id": f"sha256:{'d' * 64 if sglang else 'c' * 64}",
        "model_mount": {
            "source": (
                "volume:kairyu-qwen3-235b-nvfp4/"
                f"{gate.MODEL_VOLUME_SUBPATH}"
            ),
            "destination": gate.CHECKPOINT_ROOT,
            "read_only": True,
        },
        "gpu_device_indices": list(gate.PHYSICAL_GPU_INDICES),
        "ipc_mode": "host",
        "shm_size_bytes": 32 * 1024**3,
        "cap_add": ["SYS_NICE", "SYS_PTRACE"],
    }
    runtime = _runtime(arm, candidate)
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "arm": arm,
        "server_generation_id": f"generation-{ordinal}",
        "server_started_ns": server_started_ns,
        "container": container,
        "source": source,
        "checkpoint": gate.expected_checkpoint(),
        "checkpoint_capture_start": _checkpoint_capture("start"),
        "checkpoint_volume": {
            "name": "kairyu-qwen3-235b-nvfp4",
            "driver": "local",
            "scope": "local",
            "mountpoint": "/var/lib/docker/volumes/kairyu-qwen3-235b-nvfp4/_data",
            "subpath": gate.MODEL_VOLUME_SUBPATH,
            "rw_consumer_count": 0,
        },
        "gpus": [
            {
                "visible_index": visible,
                "physical_index": physical,
                "uuid": f"GPU-fixture-{physical}",
                "pci_bus_id": f"00000000:{physical:02X}:00.0",
                "name": "RTX PRO 6000 Blackwell",
                "total_memory_bytes": 100 * 1024**3,
            }
            for visible, physical in enumerate(gate.PHYSICAL_GPU_INDICES)
        ],
        "environment": environment,
        "runtime": runtime,
        "runtime_witness": _runtime_witness(arm, runtime),
    }


def _live_observation(
    provenance: dict[str, object],
    *,
    started_ns: int,
    completed_ns: int,
) -> dict[str, object]:
    container = provenance["container"]
    runtime = provenance["runtime"]
    assert isinstance(container, dict) and isinstance(runtime, dict)
    return gate.hashed_descriptor(
        {
            "schema_version": gate.LIVE_OBSERVATION_SCHEMA_VERSION,
            "type": "live_observation_end",
            "capture_started_ns": started_ns,
            "capture_completed_ns": completed_ns,
            "container": {
                "id": container["id"],
                "image_repo_digest": container["image_repo_digest"],
                "image_platform_digest": container["image_platform_digest"],
                "image_config_id": container["image_config_id"],
                "model_mount": container["model_mount"],
                "resolved_argv": runtime["resolved_argv"],
            },
            "source": provenance["source"],
            "checkpoint_volume": provenance["checkpoint_volume"],
            "gpus": provenance["gpus"],
            "environment": provenance["environment"],
            "runtime_witness": _runtime_witness(str(provenance["arm"]), runtime),
        }
    )


def _success_row(
    planned: gate.PlannedRequest,
    *,
    scenario_id: str,
    run_phase: str,
    arm: str,
    candidate: str,
    repeat: int | None,
    start_ns: int,
    duration_ns: int,
    ttft_ns: int,
    ordinal: int,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    end_ns = start_ns + duration_ns
    client_role = (
        "warmup" if planned.phase in {"serial_warmup", "graph_warmup"} else "measurement"
    )
    return {
        "type": "request",
        "http_client_role": client_role,
        "http_client_request_ordinal": ordinal,
        "phase": planned.phase,
        "sequence": planned.sequence,
        "attempts": 1,
        "preflight_repeat": planned.preflight_repeat,
        "graph_batch_size": planned.graph_batch_size,
        "graph_batch_sequence": planned.graph_batch_sequence,
        "open_loop_point": planned.open_loop_point,
        "rate_millirps": planned.rate_millirps,
        "prompt_text_sha256": planned.prompt_text_sha256,
        "prompt_token_ids_sha256": planned.prompt_token_ids_sha256,
        "prompt_tokens": len(planned.prompt_token_ids),
        "expected_completion_tokens": planned.expected_completion_tokens,
        "http_status": 200,
        "content_type": "text/event-stream; charset=utf-8",
        "response_request_header": f"request-{ordinal}",
        "response_id": f"completion-{scenario_id}-{ordinal}",
        "status": "success",
        "error_type": None,
        "done_events": 1,
        "terminal_choice_events": 1,
        "finish_reasons": ["length"],
        "usage_events": 1,
        "usage": {
            "prompt_tokens": len(planned.prompt_token_ids),
            "completion_tokens": planned.expected_completion_tokens,
            "cached_tokens": 0,
        },
        "timing_ns": {
            "start": start_ns,
            "first_token": start_ns + ttft_ns,
            "end": end_ns,
            "ttft": ttft_ns,
            "total": duration_ns,
        },
        "output_text_sha256": "1" * 64,
        "stream_payload_sha256": "2" * 64,
        "scenario_id": scenario_id,
        "run_phase": run_phase,
        "arm": arm,
        "candidate": candidate,
        "repeat": repeat,
        **(extra or {}),
    }


def _client_snapshot(
    *,
    role: str,
    created_ns: int,
    closed_ns: int,
    paths: list[str],
) -> dict[str, object]:
    return {
        "role": role,
        "created_ns": created_ns,
        "closed_ns": closed_ns,
        "request_count": len(paths),
        "request_path_counts": {
            path: paths.count(path) for path in sorted(set(paths))
        },
        "request_path_sequence_sha256": gate.sha256_json(paths),
    }


def _scenario_rows(
    *,
    bundle: dict[str, object],
    trace_bundle_sha256: str,
    phase: str,
    arm: str,
    candidate: str,
    repeat: int | None,
    selection: dict[str, object] | None,
    ordinal: int,
    base_ns: int,
    measurement_duration_ns: int,
    measurement_ttft_ns: int,
) -> tuple[list[dict[str, object]], int]:
    scenario_id = gate._scenario_id(
        phase=phase,
        arm=arm,
        candidate=candidate,
        repeat=repeat,
    )
    server_started_ns = base_ns + 100
    capture_start_ns = base_ns + 200
    provenance_value = _provenance(
        arm=arm,
        candidate=candidate,
        ordinal=ordinal,
        server_started_ns=server_started_ns,
    )
    provenance = gate.hashed_descriptor(provenance_value)
    graph_replays_start = 100 + ordinal * 10
    start = {
        "type": "shard_start",
        "schema_version": gate.SCHEMA_VERSION,
        "gate": gate.GATE,
        "scenario_id": scenario_id,
        "phase": phase,
        "arm": arm,
        "candidate": candidate,
        "repeat": repeat,
        "endpoint": "http://127.0.0.1:30000",
        "trace_bundle_sha256": trace_bundle_sha256,
        "benchmark": bundle["benchmark"],
        "trace": bundle["trace"],
        "graph_warmup": bundle["graph_warmup"],
        "selection": selection,
        "provenance_start": provenance,
        "kairyu_graph_witness_start": (
            _graph_witness(replays=graph_replays_start) if arm == "kairyu" else None
        ),
        "capture_start_ns": capture_start_ns,
    }
    rows: list[dict[str, object]] = [start]
    warmup_client_created_ns = capture_start_ns + 500
    cursor = capture_start_ns + 1_000
    request_ordinal = 1
    for planned in gate._serial_warmup_plan(bundle["trace"]):
        rows.append(
            _success_row(
                planned,
                scenario_id=scenario_id,
                run_phase=phase,
                arm=arm,
                candidate=candidate,
                repeat=repeat,
                start_ns=cursor,
                duration_ns=100,
                ttft_ns=10,
                ordinal=request_ordinal,
            )
        )
        request_ordinal += 1
        cursor += 200
    graph_plan = gate._graph_warmup_plan(bundle["graph_warmup"])
    offset = 0
    for batch_size in gate.GRAPH_WARMUP_BATCH_SIZES:
        release = cursor
        for planned in graph_plan[offset : offset + batch_size]:
            rows.append(
                _success_row(
                    planned,
                    scenario_id=scenario_id,
                    run_phase=phase,
                    arm=arm,
                    candidate=candidate,
                    repeat=repeat,
                    start_ns=release + 10,
                    duration_ns=100,
                    ttft_ns=10,
                    ordinal=request_ordinal,
                    extra={"graph_release_ns": release},
                )
            )
            request_ordinal += 1
        offset += batch_size
        cursor = release + 210
    warmup_client_closed_ns = cursor
    measurement_client_created_ns = warmup_client_closed_ns + 1
    cursor = measurement_client_created_ns + 100
    request_ordinal = 0
    groups: list[list[gate.PlannedRequest]]
    if phase == "preflight":
        groups = gate._preflight_plans(bundle["trace"])
    elif phase == "formal":
        groups = [gate._formal_plan(bundle["trace"])]
    else:
        assert selection is not None and isinstance(selection["value"], dict)
        rates = selection["value"]["open_loop_rates_millirps"]
        assert isinstance(rates, list)
        groups = gate._open_loop_plans(bundle["trace"], rates)
    measurement_groups: list[dict[str, int]] = []
    for group_number, group in enumerate(groups):
        group_rows: list[dict[str, object]] = []
        if phase == "open-loop":
            rate = group[0].rate_millirps
            assert isinstance(rate, int)
            period_ns = 1_000_000_000_000 // rate
            anchor = cursor
            for position, planned in enumerate(group):
                scheduled = anchor + position * period_ns
                group_rows.append(
                    _success_row(
                        planned,
                        scenario_id=scenario_id,
                        run_phase=phase,
                        arm=arm,
                        candidate=candidate,
                        repeat=repeat,
                        start_ns=scheduled + 10,
                        duration_ns=measurement_duration_ns,
                        ttft_ns=measurement_ttft_ns,
                        ordinal=request_ordinal,
                        extra={
                            "scheduled_start_ns": scheduled,
                            "scheduled_offset_ns": position * period_ns,
                            "release_lag_ns": 10,
                        },
                    )
                )
                request_ordinal += 1
        else:
            release = cursor
            for planned in group:
                group_rows.append(
                    _success_row(
                        planned,
                        scenario_id=scenario_id,
                        run_phase=phase,
                        arm=arm,
                        candidate=candidate,
                        repeat=repeat,
                        start_ns=release + 10,
                        duration_ns=measurement_duration_ns,
                        ttft_ns=measurement_ttft_ns,
                        ordinal=request_ordinal,
                        extra={"burst_release_ns": release},
                    )
                )
                request_ordinal += 1
        rows.extend(group_rows)
        starts = [int(row["timing_ns"]["start"]) for row in group_rows]
        ends = [int(row["timing_ns"]["end"]) for row in group_rows]
        measurement_groups.append({"group": group_number, "start": min(starts), "end": max(ends)})
        cursor = max(ends) + 100
    measurement_client_closed_ns = cursor
    live_observation_started_ns = cursor + 10
    live_observation_completed_ns = cursor + 20
    capture_end_ns = cursor + 100
    rows.append(
        {
            "type": "shard_end",
            "scenario_id": scenario_id,
            "phase": phase,
            "arm": arm,
            "candidate": candidate,
            "repeat": repeat,
            "serial_warmup_request_count": gate.SERIAL_WARMUP_REQUESTS,
            "graph_warmup_request_count": gate.GRAPH_WARMUP_REQUESTS,
            "measurement_request_count": sum(len(group) for group in groups),
            "measurement_groups": measurement_groups,
            "http_client_lifecycle": {
                "schema_version": gate.HTTP_CLIENT_LIFECYCLE_SCHEMA_VERSION,
                "distinct_client_instances": True,
                "measurement_requests_before_group": 0,
                "warmup_client": _client_snapshot(
                    role="warmup",
                    created_ns=warmup_client_created_ns,
                    closed_ns=warmup_client_closed_ns,
                    paths=[
                        "/v1/models",
                        *["/v1/completions"]
                        * (gate.SERIAL_WARMUP_REQUESTS + gate.GRAPH_WARMUP_REQUESTS),
                        *(["/backends"] if arm == "kairyu" else []),
                    ],
                ),
                "measurement_client": _client_snapshot(
                    role="measurement",
                    created_ns=measurement_client_created_ns,
                    closed_ns=measurement_client_closed_ns,
                    paths=[
                        *["/v1/completions"] * sum(len(group) for group in groups),
                        *(["/backends"] if arm == "kairyu" else []),
                    ],
                ),
            },
            "provenance_end": _live_observation(
                provenance_value,
                started_ns=live_observation_started_ns,
                completed_ns=live_observation_completed_ns,
            ),
            "kairyu_graph_witness_end": (
                _graph_witness(replays=graph_replays_start + 1) if arm == "kairyu" else None
            ),
            "capture_end_ns": capture_end_ns,
        }
    )
    return rows, capture_end_ns


@pytest.fixture(scope="module")
def formal_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("g4-ma3")
    checkpoint_start_path = root / "checkpoint-start.json"
    checkpoint_start = _checkpoint_capture("start")
    checkpoint_start_path.write_text(
        gate.canonical_json(checkpoint_start["value"]) + "\n",
        encoding="utf-8",
    )
    trace_path = root / "trace.json"
    bundle, trace_sha256 = _trace_bundle(trace_path)
    preflight_paths: list[Path] = []
    base_ns = 1_000_000
    ordinal = 0
    durations = {
        ("kairyu", "depth-1"): 4_000_000,
        ("kairyu", "depth-2"): 2_000_000,
        ("kairyu", "depth-5"): 3_000_000,
        ("sglang", "default"): 3_000_000,
        ("sglang", "single-batch"): 5_000_000,
        ("sglang", "two-batch"): 4_000_000,
    }
    for arm in gate.ARMS:
        for candidate in gate.PREFLIGHT_CANDIDATES[arm]:
            rows, end_ns = _scenario_rows(
                bundle=bundle,
                trace_bundle_sha256=trace_sha256,
                phase="preflight",
                arm=arm,
                candidate=candidate,
                repeat=None,
                selection=None,
                ordinal=ordinal,
                base_ns=base_ns,
                measurement_duration_ns=durations[(arm, candidate)],
                measurement_ttft_ns=100_000,
            )
            path = root / f"preflight-{arm}-{candidate}.jsonl"
            gate.write_shard(path, rows)
            preflight_paths.append(path)
            base_ns = end_ns + 1_000
            ordinal += 1
    selection_path = root / "selection.json"
    selection_value = gate.write_preflight_selection(selection_path, preflight_paths)
    selection = gate.load_selection(selection_path)
    assert selection_value["selected"] == {"kairyu": "depth-5", "sglang": "default"}
    formal_paths: list[Path] = []
    for repeat, arm in gate.FORMAL_SEQUENCE:
        candidate = selection_value["selected"][arm]
        rows, end_ns = _scenario_rows(
            bundle=bundle,
            trace_bundle_sha256=trace_sha256,
            phase="formal",
            arm=arm,
            candidate=candidate,
            repeat=repeat,
            selection=selection,
            ordinal=ordinal,
            base_ns=base_ns,
            measurement_duration_ns=2_000_000,
            measurement_ttft_ns=100_000,
        )
        path = root / f"formal-r{repeat}-{arm}.jsonl"
        gate.write_shard(path, rows)
        formal_paths.append(path)
        base_ns = end_ns + 1_000
        ordinal += 1
    checkpoint_end_path = root / "checkpoint-end.json"
    checkpoint_end = _checkpoint_capture(
        "end",
        started_ns=base_ns,
        completed_ns=base_ns + 1,
    )
    checkpoint_end_path.write_text(
        gate.canonical_json(checkpoint_end["value"]) + "\n",
        encoding="utf-8",
    )
    rows = gate.assemble_rows(
        preflight_paths=preflight_paths,
        formal_paths=formal_paths,
        selection_path=selection_path,
        checkpoint_start_path=checkpoint_start_path,
        checkpoint_end_path=checkpoint_end_path,
    )
    output_dir = root / "artifact"
    manifest = gate.write_artifact(output_dir, rows)
    return {
        "root": root,
        "output_dir": output_dir,
        "manifest": manifest,
        "selection": selection_value,
    }


def test_exact_sglang_v0516_identity_and_working_argv() -> None:
    assert gate.DEFAULT_SGLANG_MOE_A2A_BACKEND == "none"
    assert gate.SGLANG_IMAGE_REPO_DIGEST.endswith(
        "984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c"
    )
    expected = (
        "sglang serve --model-path /models/qwen3-235b-nvfp4 --host 0.0.0.0 "
        "--port 30000 --tp-size 4 --dp-size 4 --ep-size 4 --enable-dp-attention "
        "--load-balance-method round_robin --dtype bfloat16 --kv-cache-dtype bfloat16 "
        "--fp4-gemm-backend flashinfer_cutlass --moe-a2a-backend none "
        "--moe-runner-backend flashinfer_cutlass --page-size 16 "
        "--max-running-requests 256 --max-total-tokens 16384 "
        "--chunked-prefill-size 8192 --max-prefill-tokens 2048 "
        "--schedule-policy fcfs --log-level-http warning "
        "--cuda-graph-max-bs-decode 32 --cuda-graph-backend-prefill disabled"
    ).split()
    assert gate.expected_sglang_argv(overlap_mode="default") == expected
    with pytest.raises(ValueError, match="pinned to 'none'"):
        gate.expected_sglang_argv(
            overlap_mode="default",
            moe_a2a_backend="flashinfer",
        )


def test_benchmark_pins_fresh_measurement_http_client_contract() -> None:
    lifecycle = gate.benchmark_config(a6.SHAREGPT_DATASET_SHA256)[
        "http_client_lifecycle"
    ]
    assert lifecycle == {
        "applies_to_arms": ["kairyu", "sglang"],
        "applies_to_phases": ["preflight", "formal"],
        "warmup_client_request_order": [
            "model_probe",
            "serial_warmup",
            "graph_warmup",
            "kairyu_graph_start_witness_if_kairyu",
        ],
        "warmup_client_fully_closed_before_measurement_client_created": True,
        "measurement_client": "distinct fresh connection pool",
        "measurement_client_requests_before_synchronized_group": 0,
        "measurement_client_request_order": [
            "one synchronized preflight_or_formal_group",
            "kairyu_graph_end_witness_if_kairyu",
        ],
        "measurement_client_fully_closed_after_witness": True,
    }


def test_tracked_http_client_records_paths_and_only_emits_evidence_after_close() -> None:
    transport = gate.httpx.MockTransport(
        lambda _request: gate.httpx.Response(200, json={"ok": True})
    )

    async def exercise() -> dict[str, object]:
        client = gate._TrackedAsyncClient(
            role="measurement",
            transport=transport,
        )
        with pytest.raises(gate.GateEvidenceError, match="not fully closed"):
            client.evidence()
        async with client:
            await client.get("http://127.0.0.1:30000/v1/models")
            async with client.stream(
                "POST",
                "http://127.0.0.1:30000/v1/completions",
            ) as response:
                await response.aread()
        return client.evidence()

    evidence = asyncio.run(exercise())
    assert evidence["role"] == "measurement"
    assert evidence["request_count"] == 2
    assert evidence["request_path_counts"] == {
        "/v1/completions": 1,
        "/v1/models": 1,
    }
    assert evidence["request_path_sequence_sha256"] == gate.sha256_json(
        ["/v1/models", "/v1/completions"]
    )
    assert int(evidence["created_ns"]) < int(evidence["closed_ns"])


def test_trace_bundle_fails_closed_without_client_lifecycle_contract(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    bundle, _digest = _trace_bundle(path)
    changed = copy.deepcopy(bundle)
    changed["benchmark"].pop("http_client_lifecycle")
    path.write_text(gate.canonical_json(changed) + "\n", encoding="utf-8")
    with pytest.raises(gate.GateEvidenceError, match="G4 M-A3 contract"):
        gate.load_trace_bundle(path, dataset_sha256=a6.SHAREGPT_DATASET_SHA256)


def test_provenance_json_capture_preserves_observation_label(monkeypatch) -> None:
    monkeypatch.setattr(gate, "_capture_command", lambda _command: "{")

    with pytest.raises(gate.GateEvidenceError, match="Docker inspect is not valid JSON"):
        gate._capture_single_object(("docker", "inspect"), label="Docker inspect")


def test_runtime_version_capture_preserves_probe_label(monkeypatch) -> None:
    monkeypatch.setattr(gate, "_capture_command", lambda _command: "{")

    with pytest.raises(
        gate.GateEvidenceError,
        match="runtime-version probe is not valid JSON",
    ):
        gate._capture_runtime_versions("server", "kairyu")


@pytest.mark.parametrize(
    "captured",
    (
        ["SYS_NICE", "SYS_PTRACE"],
        ["CAP_SYS_NICE", "CAP_SYS_PTRACE"],
        ["CAP_SYS_PTRACE", "SYS_NICE"],
    ),
)
def test_docker_capability_prefix_is_strictly_canonicalized(captured: list[str]) -> None:
    assert gate._canonical_cap_add(captured) == ["SYS_NICE", "SYS_PTRACE"]


@pytest.mark.parametrize(
    "captured",
    (
        ["SYS_NICE", "CAP_SYS_NICE"],
        ["SYS_NICE", "SYS_PTRACE", "SYS_ADMIN"],
        ["SYS_NICE", "SYS_ADMIN"],
    ),
)
def test_docker_capability_canonicalization_rejects_duplicate_extra_or_unknown(
    captured: list[str],
) -> None:
    with pytest.raises(gate.GateEvidenceError, match="capabilit"):
        gate._canonical_cap_add(captured)


def test_provenance_pins_versions_image_and_all_checkpoint_shards() -> None:
    provenance = _provenance(
        arm="sglang",
        candidate="default",
        ordinal=0,
        server_started_ns=100,
    )
    assert gate._provenance_value_valid(
        provenance,
        arm="sglang",
        candidate="default",
        moe_a2a_backend="none",
    )
    assert len(provenance["checkpoint"]["weight_shards"]) == 27
    changed = copy.deepcopy(provenance)
    changed["environment"]["flashinfer_version"] = "0.6.13"
    assert not gate._provenance_value_valid(
        changed,
        arm="sglang",
        candidate="default",
        moe_a2a_backend="none",
    )


def test_runtime_pins_kairyu_graph_and_matched_cache_capacity() -> None:
    kairyu = _runtime("kairyu", "depth-5")
    sglang = _runtime("sglang", "default")
    assert kairyu["decode_cuda_graph"] is True
    assert sglang["decode_cuda_graph"] is True
    assert gate._runtime_valid(
        kairyu,
        arm="kairyu",
        candidate="depth-5",
        moe_a2a_backend="none",
    )
    assert gate._runtime_valid(
        sglang,
        arm="sglang",
        candidate="default",
        moe_a2a_backend="none",
    )
    assert kairyu["max_total_tokens"] == sglang["max_total_tokens"] == 65_536
    assert (
        kairyu["max_total_tokens_per_owner"]
        == sglang["max_total_tokens_per_owner"]
        == 16_384
    )
    assert kairyu["configured_max_prefill_tokens"] == 8_192
    assert sglang["configured_max_prefill_tokens"] == 2_048
    assert (
        kairyu["resolved_max_prefill_tokens_per_owner"]
        == sglang["resolved_max_prefill_tokens_per_owner"]
        == 2_048
    )
    changed = copy.deepcopy(kairyu)
    changed["cuda_graph_max_pages"] = gate.KAIRYU_CUDA_GRAPH_MAX_PAGES + 1
    assert not gate._runtime_valid(
        changed,
        arm="kairyu",
        candidate="depth-5",
        moe_a2a_backend="none",
    )
    for field, invalid in (
        ("moe_a2a_backend", "torch.distributed:nccl"),
        ("moe_runner_backend", gate.SGLANG_MOE_RUNNER_BACKEND),
        ("scheduler_policy", "priority"),
    ):
        changed = copy.deepcopy(kairyu)
        changed[field] = invalid
        assert not gate._runtime_valid(
            changed,
            arm="kairyu",
            candidate="depth-5",
            moe_a2a_backend="none",
        )


@pytest.mark.parametrize(
    ("arm", "candidate", "changed_field"),
    (
        ("kairyu", "depth-5", "pipeline_depth"),
        ("sglang", "default", "chunked_prefill_size"),
    ),
)
def test_live_runtime_witness_must_match_static_runtime(
    arm: str,
    candidate: str,
    changed_field: str,
) -> None:
    runtime = _runtime(arm, candidate)
    witness = _runtime_witness(arm, runtime)
    assert gate._runtime_witness_valid(witness, arm=arm, runtime=runtime)

    changed = copy.deepcopy(witness)
    response = changed["response"]
    assert isinstance(response, dict)
    if arm == "kairyu":
        engines = response["engines"]
        assert isinstance(engines, list) and isinstance(engines[0], dict)
        engines[0][changed_field] = 4
    else:
        response[changed_field] = gate.RESOLVED_PREFILL_TOKENS_PER_OWNER + 1
    assert not gate._runtime_witness_valid(changed, arm=arm, runtime=runtime)


def test_capture_checkpoint_binds_running_container_volume_source_and_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "checkpoint-start.json"
    expected = gate.expected_checkpoint()
    container_id = "1" * 64
    image_config_id = f"sha256:{'2' * 64}"
    image_repo_digest = f"ghcr.io/ytworks/kairyu@sha256:{'3' * 64}"
    model_volume = "kairyu-qwen3-235b-nvfp4"
    container = {
        "Id": container_id,
        "Image": image_config_id,
        "Path": "sleep",
        "Args": ["infinity"],
        "State": {"Running": True},
        "Config": {"Image": image_repo_digest, "WorkingDir": gate.CAPTURE_SOURCE_ROOT},
        "HostConfig": {
            "NetworkMode": "none",
            "DeviceRequests": None,
            "PortBindings": None,
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": str(source_root.resolve()),
                    "Target": gate.CAPTURE_SOURCE_ROOT,
                    "ReadOnly": True,
                },
                {
                    "Type": "volume",
                    "Source": model_volume,
                    "Target": gate.CHECKPOINT_ROOT,
                    "ReadOnly": True,
                    "VolumeOptions": {"Subpath": gate.MODEL_VOLUME_SUBPATH},
                },
            ]
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": str(source_root.resolve()),
                "Destination": gate.CAPTURE_SOURCE_ROOT,
                "RW": False,
            },
            {
                "Type": "volume",
                "Name": model_volume,
                "Destination": gate.CHECKPOINT_ROOT,
                "RW": False,
            },
        ],
    }
    helper_checkpoint: object = expected

    def fake_capture(command: object) -> str:
        nonlocal helper_checkpoint
        observed = tuple(command)
        if observed == ("docker", "inspect", "checkpoint-capture"):
            return json.dumps([container])
        if observed == ("docker", "image", "inspect", image_repo_digest):
            return json.dumps(
                [{"Id": image_config_id, "RepoDigests": [image_repo_digest]}]
            )
        if observed == ("git", "-C", str(source_root.resolve()), "rev-parse", "HEAD"):
            return "e" * 40 + "\n"
        if observed[:4] == ("git", "-C", str(source_root.resolve()), "status"):
            return ""
        if observed[:4] == ("git", "-C", str(source_root.resolve()), "ls-files"):
            return (
                "verification/l1/correctness/g4_ma1_qwen3_235b_nvfp4_capture.py\n"
                "verification/l1/performance/g4_ma3_sglang_bench.py\n"
            )
        if observed == ("docker", "volume", "inspect", model_volume):
            return json.dumps(
                [
                    {
                        "Name": model_volume,
                        "Driver": "local",
                        "Scope": "local",
                        "Mountpoint": f"/var/lib/docker/volumes/{model_volume}/_data",
                    }
                ]
            )
        if observed == (
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"volume={model_volume}",
        ):
            return container_id + "\n"
        if observed == ("docker", "inspect", container_id):
            return json.dumps([container])
        if observed == (
            "docker",
            "exec",
            container_id,
            gate.CAPTURE_PYTHON,
            gate.CAPTURE_HELPER,
            gate._CHECKPOINT_HELPER_COMMAND,
        ):
            return gate.canonical_json(helper_checkpoint) + "\n"
        raise AssertionError(f"unexpected checkpoint observation: {observed!r}")

    timestamps = iter((10, 20))
    monkeypatch.setattr(gate.time, "perf_counter_ns", lambda: next(timestamps))
    monkeypatch.setattr(gate, "_capture_command", fake_capture)

    captured = gate.capture_checkpoint(
        output,
        phase="start",
        container_name="checkpoint-capture",
        source_root=source_root,
    )
    assert captured["checkpoint"] == gate.hashed_descriptor(expected)
    assert captured["capture_container"]["model_mount"]["source"] == (
        f"volume:{model_volume}/{gate.MODEL_VOLUME_SUBPATH}"
    )
    assert gate.load_checkpoint_capture(output, phase="start")["value"] == captured

    changed = copy.deepcopy(expected)
    changed["config_sha256"] = "0" * 64
    helper_checkpoint = changed
    timestamps = iter((30, 40))
    with pytest.raises(gate.GateEvidenceError, match="differs from the pinned"):
        gate.capture_checkpoint(
            tmp_path / "checkpoint-end.json",
            phase="end",
            container_name="checkpoint-capture",
            source_root=source_root,
        )


def test_backends_fetch_retains_only_the_pinned_engine_witness() -> None:
    expected = _graph_witness(replays=19)
    response = _BackendsResponse(
        200,
        {
            "engines": [
                {"model": "unrelated"},
                {**expected, "unrelated_diagnostic": "ignored"},
            ]
        },
    )
    client = _BackendsClient(response)
    observed = asyncio.run(
        gate._fetch_kairyu_graph_witness(client, "http://127.0.0.1:30000")
    )
    assert observed == expected
    assert client.urls == ["http://127.0.0.1:30000/backends"]


@pytest.mark.parametrize(
    ("response", "message"),
    (
        (_BackendsResponse(503, {}), "HTTP 503"),
        (_BackendsResponse(200, {"engines": []}), "exactly one"),
        (
            _BackendsResponse(200, {"engines": [{"model": gate.SERVED_MODEL_NAME}]}),
            "incomplete or malformed",
        ),
    ),
)
def test_backends_fetch_fails_immediately_on_http_or_shape(
    response: _BackendsResponse,
    message: str,
) -> None:
    client = _BackendsClient(response)
    with pytest.raises(gate.GateEvidenceError, match=message):
        asyncio.run(
            gate._fetch_kairyu_graph_witness(
                client,
                "http://127.0.0.1:30000",
            )
        )


def test_full_raw_replay_passes_exact_boundary(formal_artifact: dict[str, object]) -> None:
    output_dir = formal_artifact["output_dir"]
    assert isinstance(output_dir, Path)
    verified = gate.verify_artifact(output_dir, assert_gate=True)
    replayed = gate.replay_artifact(output_dir, assert_gate=True)
    assert verified == replayed
    assert verified["passed"] is True
    medians = verified["comparisons"]["paired_median"]
    assert gate._fraction_from_record(medians["completion_tok_s_per_gpu_kairyu_over_sglang"]) == 1
    assert gate._fraction_from_record(medians["ttft_p99_kairyu_over_sglang"]) == 1


@pytest.mark.parametrize(
    "tamper",
    (
        "prior-request",
        "pool-overlap",
        "request-role",
        "request-ordinal",
        "warmup-phase-ordinal-swap",
        "request-sequence",
    ),
)
def test_raw_replay_rejects_http_client_lifecycle_tamper(
    formal_artifact: dict[str, object],
    tmp_path: Path,
    tamper: str,
) -> None:
    output_dir = formal_artifact["output_dir"]
    assert isinstance(output_dir, Path)
    rows = copy.deepcopy(gate._read_jsonl(output_dir / gate.RAW_NAME))
    start = next(
        row
        for row in rows
        if row.get("type") == "scenario_start"
        and row.get("scenario_id") == "formal-r0-kairyu"
    )
    end = next(
        row
        for row in rows
        if row.get("type") == "scenario_end"
        and row.get("scenario_id") == start["scenario_id"]
    )
    lifecycle = end["http_client_lifecycle"]
    assert isinstance(lifecycle, dict)
    warmup = lifecycle["warmup_client"]
    measurement = lifecycle["measurement_client"]
    assert isinstance(warmup, dict) and isinstance(measurement, dict)
    if tamper == "prior-request":
        lifecycle["measurement_requests_before_group"] = 1
    elif tamper == "pool-overlap":
        warmup["closed_ns"] = measurement["created_ns"]
    elif tamper == "request-role":
        request = next(
            row
            for row in rows
            if row.get("type") == "request"
            and row.get("scenario_id") == start["scenario_id"]
            and row.get("phase") == "formal"
        )
        request["http_client_role"] = "warmup"
    elif tamper == "request-ordinal":
        request = next(
            row
            for row in rows
            if row.get("type") == "request"
            and row.get("scenario_id") == start["scenario_id"]
            and row.get("phase") == "formal"
        )
        request["http_client_request_ordinal"] = 1
    elif tamper == "warmup-phase-ordinal-swap":
        serial = next(
            row
            for row in rows
            if row.get("type") == "request"
            and row.get("scenario_id") == start["scenario_id"]
            and row.get("phase") == "serial_warmup"
        )
        graph = next(
            row
            for row in rows
            if row.get("type") == "request"
            and row.get("scenario_id") == start["scenario_id"]
            and row.get("phase") == "graph_warmup"
        )
        serial["http_client_request_ordinal"], graph["http_client_request_ordinal"] = (
            graph["http_client_request_ordinal"],
            serial["http_client_request_ordinal"],
        )
    else:
        measurement["request_path_sequence_sha256"] = "0" * 64
    with pytest.raises(gate.GateEvidenceError, match="HTTP client lifecycle"):
        gate.write_artifact(tmp_path / tamper, rows)


@pytest.mark.parametrize("tamper", ("endpoint", "end-container", "end-gpu-exclusive"))
def test_raw_replay_rejects_endpoint_or_end_live_observation_tamper(
    formal_artifact: dict[str, object],
    tmp_path: Path,
    tamper: str,
) -> None:
    output_dir = formal_artifact["output_dir"]
    assert isinstance(output_dir, Path)
    rows = copy.deepcopy(gate._read_jsonl(output_dir / gate.RAW_NAME))
    start = next(row for row in rows if row.get("type") == "scenario_start")
    end = next(
        row
        for row in rows
        if row.get("type") == "scenario_end" and row.get("scenario_id") == start["scenario_id"]
    )
    if tamper == "endpoint":
        start["endpoint"] = "http://127.0.0.1:30001"
    else:
        descriptor = end["provenance_end"]
        assert isinstance(descriptor, dict) and isinstance(descriptor["value"], dict)
        changed = copy.deepcopy(descriptor["value"])
        if tamper == "end-container":
            changed["container"]["id"] = "f" * 64
        else:
            changed["environment"]["gpu_jobs_exclusive"] = False
        end["provenance_end"] = gate.hashed_descriptor(changed)
    with pytest.raises(gate.GateEvidenceError):
        gate.write_artifact(tmp_path / tamper, rows)


def test_preflight_winner_is_raw_derived(
    formal_artifact: dict[str, object],
) -> None:
    selection = formal_artifact["selection"]
    assert selection["selected"] == {"kairyu": "depth-5", "sglang": "default"}
    assert len(selection["preflight_raw_sha256"]) == 2


def test_manifest_tamper_is_rejected_but_raw_only_replay_ignores_it(
    formal_artifact: dict[str, object],
) -> None:
    output_dir = formal_artifact["output_dir"]
    assert isinstance(output_dir, Path)
    manifest_path = output_dir / gate.MANIFEST_NAME
    original = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original)
    manifest["passed"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        with pytest.raises(gate.GateEvidenceError, match="manifest differs"):
            gate.verify_artifact(output_dir)
        assert gate.replay_artifact(output_dir, assert_gate=True)["passed"] is True
    finally:
        manifest_path.write_text(original, encoding="utf-8")


def test_binding_retry_is_not_excluded(formal_artifact: dict[str, object], tmp_path: Path) -> None:
    output_dir = formal_artifact["output_dir"]
    assert isinstance(output_dir, Path)
    rows = copy.deepcopy(gate._read_jsonl(output_dir / gate.RAW_NAME))
    request = next(
        row for row in rows if row.get("type") == "request" and row.get("phase") == "formal"
    )
    request["attempts"] = 2
    manifest = gate.write_artifact(tmp_path / "retry", rows)
    assert manifest["passed"] is False
    assert (
        manifest["checks"]["binding"][
            "strict_streaming_binding_requests_all_successful_exactly_once_no_retry"
        ]
        is False
    )


@pytest.mark.parametrize(
    "tamper",
    (
        "end-checkpoint",
        "end-source",
        "end-volume",
        "generation-start",
        "end-timeline",
    ),
)
def test_checkpoint_matrix_tamper_fails_raw_replay_binding(
    formal_artifact: dict[str, object],
    tmp_path: Path,
    tamper: str,
) -> None:
    output_dir = formal_artifact["output_dir"]
    assert isinstance(output_dir, Path)
    rows = copy.deepcopy(gate._read_jsonl(output_dir / gate.RAW_NAME))
    if tamper == "end-checkpoint":
        checkpoint_end = rows[-1]["checkpoint_end"]
        assert isinstance(checkpoint_end, dict)
        checkpoint = checkpoint_end["value"]["checkpoint"]
        assert isinstance(checkpoint, dict) and isinstance(checkpoint["value"], dict)
        checkpoint["value"]["config_sha256"] = "0" * 64
    elif tamper in {"end-source", "end-volume"}:
        checkpoint_end = rows[-1]["checkpoint_end"]
        assert isinstance(checkpoint_end, dict) and isinstance(checkpoint_end["value"], dict)
        changed = copy.deepcopy(checkpoint_end["value"])
        if tamper == "end-source":
            changed["source"]["commit"] = "f" * 40
        else:
            changed["checkpoint_volume"]["name"] = "different-model-volume"
            changed["capture_container"]["model_mount"]["source"] = (
                f"volume:different-model-volume/{gate.MODEL_VOLUME_SUBPATH}"
            )
        rows[-1]["checkpoint_end"] = gate.hashed_descriptor(changed)
    elif tamper == "generation-start":
        start = next(
            row
            for row in rows
            if row.get("type") == "scenario_start"
            and row.get("phase") == "formal"
            and row.get("arm") == "kairyu"
        )
        provenance = start["provenance_start"]
        assert isinstance(provenance, dict) and isinstance(provenance["value"], dict)
        changed = copy.deepcopy(provenance["value"])
        changed["checkpoint_capture_start"] = _checkpoint_capture(
            "start", started_ns=1, completed_ns=3
        )
        descriptor = gate.hashed_descriptor(changed)
        start["provenance_start"] = descriptor
    else:
        latest_capture_end = max(
            int(row["capture_end_ns"])
            for row in rows
            if row.get("type") == "scenario_end"
        )
        checkpoint_end = rows[-1]["checkpoint_end"]
        assert isinstance(checkpoint_end, dict) and isinstance(checkpoint_end["value"], dict)
        changed = copy.deepcopy(checkpoint_end["value"])
        changed["capture_started_ns"] = latest_capture_end
        rows[-1]["checkpoint_end"] = gate.hashed_descriptor(changed)

    manifest = gate.write_artifact(tmp_path / tamper, rows)
    assert manifest["passed"] is False
    assert (
        manifest["checks"]["binding"][
            "checkpoint_bytes_hashed_before_and_after_full_matrix_exact"
        ]
        is False
    )


@pytest.mark.parametrize(
    "tamper",
    (
        "direct-disabled",
        "missing-bucket",
        "capture-increased",
        "fallback-increased",
        "replay-not-increased",
    ),
)
def test_kairyu_dynamic_graph_witness_tamper_fails_binding(
    formal_artifact: dict[str, object],
    tmp_path: Path,
    tamper: str,
) -> None:
    output_dir = formal_artifact["output_dir"]
    assert isinstance(output_dir, Path)
    rows = copy.deepcopy(gate._read_jsonl(output_dir / gate.RAW_NAME))
    start = next(
        row
        for row in rows
        if row.get("type") == "scenario_start"
        and row.get("phase") == "formal"
        and row.get("arm") == "kairyu"
    )
    end = next(
        row
        for row in rows
        if row.get("type") == "scenario_end" and row.get("scenario_id") == start["scenario_id"]
    )
    start_witness = start["kairyu_graph_witness_start"]
    end_witness = end["kairyu_graph_witness_end"]
    assert isinstance(start_witness, dict) and isinstance(end_witness, dict)
    if tamper == "direct-disabled":
        transport = start_witness["moe_collective_transport"]
        assert isinstance(transport, dict)
        transport["direct_nccl_active"] = False
    elif tamper == "missing-bucket":
        buckets = start_witness["cuda_graph_buckets"]
        assert isinstance(buckets, list)
        buckets.pop()
    elif tamper == "capture-increased":
        end_witness["cuda_graph_captures"] = int(end_witness["cuda_graph_captures"]) + 1
    elif tamper == "fallback-increased":
        end_witness["cuda_graph_eager_fallbacks"] = (
            int(end_witness["cuda_graph_eager_fallbacks"]) + 1
        )
    else:
        end_witness["cuda_graph_replays"] = start_witness["cuda_graph_replays"]

    manifest = gate.write_artifact(tmp_path / tamper, rows)
    assert manifest["passed"] is False
    assert (
        manifest["checks"]["binding"][
            "kairyu_request_owned_ep4_direct_nccl_and_cuda_graph_replay_exact"
        ]
        is False
    )


def test_unknown_raw_request_field_fails_closed(
    formal_artifact: dict[str, object], tmp_path: Path
) -> None:
    output_dir = formal_artifact["output_dir"]
    assert isinstance(output_dir, Path)
    rows = copy.deepcopy(gate._read_jsonl(output_dir / gate.RAW_NAME))
    request = next(row for row in rows if row.get("type") == "request")
    request["unexpected"] = True
    with pytest.raises(gate.GateEvidenceError, match="unknown"):
        gate.write_artifact(tmp_path / "unknown", rows)


@pytest.mark.parametrize(
    "tamper",
    (
        "request-timing",
        "burst-release",
        "graph-release",
        "measurement-group",
    ),
)
def test_scenario_timestamps_must_belong_to_the_capture_interval(
    formal_artifact: dict[str, object],
    tmp_path: Path,
    tamper: str,
) -> None:
    output_dir = formal_artifact["output_dir"]
    assert isinstance(output_dir, Path)
    rows = copy.deepcopy(gate._read_jsonl(output_dir / gate.RAW_NAME))
    start = next(
        row
        for row in rows
        if row.get("type") == "scenario_start"
        and row.get("phase") == "formal"
        and row.get("arm") == "kairyu"
    )
    capture_start = start["capture_start_ns"]
    assert isinstance(capture_start, int)
    scenario_id = start["scenario_id"]
    scenario_rows = [row for row in rows if row.get("scenario_id") == scenario_id]
    if tamper == "request-timing":
        request = next(row for row in scenario_rows if row.get("type") == "request")
        timing = request["timing_ns"]
        assert isinstance(timing, dict)
        shift = int(timing["start"]) - capture_start + 1
        for key in ("start", "first_token", "end"):
            timing[key] = int(timing[key]) - shift
    elif tamper == "measurement-group":
        end = next(row for row in scenario_rows if row.get("type") == "scenario_end")
        groups = end["measurement_groups"]
        assert isinstance(groups, list) and isinstance(groups[0], dict)
        capture_end = end["capture_end_ns"]
        assert isinstance(capture_end, int)
        groups[0]["end"] = capture_end + 1
    else:
        field = {
            "burst-release": "burst_release_ns",
            "graph-release": "graph_release_ns",
        }[tamper]
        request = next(row for row in scenario_rows if field in row)
        request[field] = capture_start - 1
    with pytest.raises(gate.GateEvidenceError, match="capture interval"):
        gate.write_artifact(tmp_path / tamper, rows)


def _mock_capture_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    arm: str,
    foreign_pid: bool = False,
    dirty_source: bool = False,
    rw_volume_consumer: bool = False,
    model_subpath: str = gate.MODEL_VOLUME_SUBPATH,
) -> tuple[Path | None, Path, Path]:
    container_name = f"formal-{arm}"
    model_volume = "kairyu-qwen3-235b-nvfp4"
    checkpoint_start_path = tmp_path / "checkpoint-start.json"
    checkpoint_start_path.write_text(
        gate.canonical_json(_checkpoint_capture("start")["value"]) + "\n",
        encoding="utf-8",
    )
    source_root = tmp_path / "source" if arm == "kairyu" else None
    if source_root is not None:
        source_root.mkdir()
    commit = "e" * 40
    config_id = f"sha256:{'b' * 64 if arm == 'kairyu' else 'c' * 64}"
    repo_digest = (
        f"127.0.0.1:5168/kairyu-g4-ma3@sha256:{'a' * 64}"
        if arm == "kairyu"
        else gate.SGLANG_IMAGE_REPO_DIGEST
    )
    env = ["CUDA_VISIBLE_DEVICES=0,1,2,3"]
    mounts: list[dict[str, object]] = [
        {
            "Type": "volume",
            "Name": model_volume,
            "Source": f"/var/lib/docker/volumes/{model_volume}/_data",
            "Destination": gate.CHECKPOINT_ROOT,
            "RW": False,
        }
    ]
    if source_root is not None:
        env.extend(
            [
                "PYTHONPATH=/workspace",
                "GIT_CONFIG_COUNT=1",
                "GIT_CONFIG_KEY_0=safe.directory",
                "GIT_CONFIG_VALUE_0=/workspace",
            ]
        )
        mounts.append(
            {
                "Type": "bind",
                "Source": str(source_root.resolve()),
                "Destination": "/workspace",
                "RW": False,
            }
        )
    resolved = (
        [
            "/app/.venv/bin/python",
            "/workspace/verification/l1/performance/g4_ma3_kairyu_server.py",
            "--pipeline-depth",
            "5",
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
        ]
        if arm == "kairyu"
        else gate.expected_sglang_argv(overlap_mode="default")
    )
    container = {
        "Id": "1" * 64,
        "Image": config_id,
        "Path": resolved[0],
        "Args": resolved[1:],
        "State": {"Running": True},
        "Config": {
            "Image": repo_digest,
            "Env": env,
            "WorkingDir": "/workspace" if arm == "kairyu" else "/sgl-workspace",
        },
        "HostConfig": {
            "IpcMode": "host",
            "ShmSize": 32 * 1024**3,
            "CapAdd": ["CAP_SYS_NICE", "CAP_SYS_PTRACE"],
            "DeviceRequests": [
                {
                    "Driver": "nvidia",
                    "DeviceIDs": ["4", "5", "6", "7"],
                    "Capabilities": [["gpu"]],
                }
            ],
            "PortBindings": {
                "30000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "30000"}]
            },
            "Mounts": [
                {
                    "Type": "volume",
                    "Source": model_volume,
                    "Target": gate.CHECKPOINT_ROOT,
                    "ReadOnly": True,
                    "VolumeOptions": {"Subpath": model_subpath},
                }
            ],
        },
        "Mounts": mounts,
    }
    image = {
        "Id": config_id,
        "Os": "linux",
        "Architecture": "amd64",
        "RepoDigests": [repo_digest],
        "Config": {
            "Labels": (
                {"org.opencontainers.image.revision": commit}
                if arm == "kairyu"
                else {}
            )
        },
    }
    rw_consumer = copy.deepcopy(container)
    rw_consumer["Id"] = "2" * 64
    rw_consumer["Mounts"][0]["RW"] = True
    rw_consumer["HostConfig"]["Mounts"][0]["ReadOnly"] = False
    host_rows = [
        f"{physical}, GPU-formal-{physical}, 00000000:{physical:02X}:00.0, "
        "NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84, 97887"
        for physical in gate.PHYSICAL_GPU_INDICES
    ]
    container_rows = [
        f"{visible}, GPU-formal-{physical}, 00000000:{physical:02X}:00.0, "
        "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887"
        for visible, physical in enumerate(gate.PHYSICAL_GPU_INDICES)
    ]
    pids = [100, 101, 102, 103]
    app_rows = [
        f"GPU-formal-{physical}, {999 if foreign_pid and visible == 0 else pids[visible]}"
        for visible, physical in enumerate(gate.PHYSICAL_GPU_INDICES)
    ]
    versions = {
        "cuda": gate.SGLANG_CUDA_VERSION,
        "nccl": [2, 29, 7] if arm == "kairyu" else [2, 28, 9],
        "torch": "2.12.1+cu130" if arm == "kairyu" else gate.SGLANG_TORCH_VERSION,
        "flashinfer": gate.SGLANG_FLASHINFER_VERSION,
        "engine": "0.1.0" if arm == "kairyu" else gate.SGLANG_VERSION,
    }

    def fake_capture(command: object) -> str:
        observed = tuple(command)
        if observed == ("docker", "inspect", container_name):
            return json.dumps([container])
        if observed == ("docker", "image", "inspect", repo_digest):
            return json.dumps([image])
        if observed == ("docker", "volume", "inspect", model_volume):
            return json.dumps(
                [
                    {
                        "Name": model_volume,
                        "Driver": "local",
                        "Scope": "local",
                        "Mountpoint": f"/var/lib/docker/volumes/{model_volume}/_data",
                    }
                ]
            )
        if observed == (
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"volume={model_volume}",
        ):
            ids = [str(container["Id"])]
            if rw_volume_consumer:
                ids.append(str(rw_consumer["Id"]))
            return "\n".join(ids) + "\n"
        expected_consumer_ids = [str(container["Id"])]
        expected_consumers = [container]
        if rw_volume_consumer:
            expected_consumer_ids.append(str(rw_consumer["Id"]))
            expected_consumers.append(rw_consumer)
        if observed == ("docker", "inspect", *expected_consumer_ids):
            return json.dumps(expected_consumers)
        if observed[:2] == ("docker", "exec") and observed[2] in {
            container_name,
            str(container["Id"]),
        }:
            if "nvidia-smi" in observed:
                return "\n".join(container_rows) + "\n"
            return json.dumps(versions) + "\n"
        if observed in {
            ("docker", "top", container_name, "-eo", "pid"),
            ("docker", "top", str(container["Id"]), "-eo", "pid"),
        }:
            return "PID\n" + "\n".join(str(pid) for pid in pids) + "\n"
        if observed[:2] == (
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total",
        ):
            return "\n".join(host_rows) + "\n"
        if observed[:2] == ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid"):
            return "\n".join(app_rows) + "\n"
        if observed[:3] == ("git", "-C", str(source_root.resolve())):
            if observed[3:5] == ("rev-parse", "HEAD"):
                return commit + "\n"
            if observed[3] == "status":
                return " M kairyu/models/moe_parallel.py\n" if dirty_source else ""
            if observed[3] == "ls-files":
                return "verification/l1/performance/g4_ma3_kairyu_server.py\n"
        raise AssertionError(f"unexpected provenance observation: {observed!r}")

    monkeypatch.setattr(gate, "_capture_command", fake_capture)
    candidate = "depth-5" if arm == "kairyu" else "default"
    monkeypatch.setattr(
        gate,
        "_capture_runtime_witness",
        lambda observed_arm: _runtime_witness(observed_arm, _runtime(arm, candidate)),
    )
    monkeypatch.setattr(gate.socket, "gethostname", lambda: "formal-host")
    return source_root, checkpoint_start_path, tmp_path / f"{arm}-provenance.json"


@pytest.mark.parametrize(("arm", "candidate"), (("kairyu", "depth-5"), ("sglang", "default")))
def test_capture_provenance_uses_live_container_gpu_and_version_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arm: str,
    candidate: str,
) -> None:
    source_root, checkpoint_start, output = _mock_capture_observations(
        monkeypatch,
        tmp_path,
        arm=arm,
    )
    captured = gate.capture_provenance(
        output,
        arm=arm,
        candidate=candidate,
        container_name=f"formal-{arm}",
        server_generation_id=f"generation-{arm}",
        server_started_ns=123_456,
        model_volume="kairyu-qwen3-235b-nvfp4",
        checkpoint_start_path=checkpoint_start,
        source_root=source_root,
    )
    assert captured["environment"]["host_id"] == "formal-host"
    assert captured["container"]["model_mount"]["source"].endswith(
        f"/{gate.MODEL_VOLUME_SUBPATH}"
    )
    assert [row["physical_index"] for row in captured["gpus"]] == [4, 5, 6, 7]
    assert gate.load_provenance(
        output,
        arm=arm,
        candidate=candidate,
        moe_a2a_backend=gate.DEFAULT_SGLANG_MOE_A2A_BACKEND,
    )["value"] == captured
    live_end = gate.capture_live_observation(captured)
    assert live_end["container"]["id"] == captured["container"]["id"]
    assert live_end["checkpoint_volume"] == captured["checkpoint_volume"]
    assert gate._live_observation_value_valid(live_end, provenance=captured)


@pytest.mark.parametrize(
    ("capture_change", "message"),
    (
        ({"foreign_pid": True}, "foreign compute process"),
        ({"dirty_source": True}, "clean tracked commit"),
        ({"rw_volume_consumer": True}, "read-write consumer"),
        ({"model_subpath": "wrong"}, "volume-subpath"),
    ),
)
def test_capture_provenance_rejects_unowned_gpu_dirty_source_or_wrong_subpath(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capture_change: dict[str, object],
    message: str,
) -> None:
    source_root, checkpoint_start, output = _mock_capture_observations(
        monkeypatch,
        tmp_path,
        arm="kairyu",
        **capture_change,
    )
    with pytest.raises(gate.GateEvidenceError, match=message):
        gate.capture_provenance(
            output,
            arm="kairyu",
            candidate="depth-5",
            container_name="formal-kairyu",
            server_generation_id="generation-kairyu",
                server_started_ns=123_456,
                model_volume="kairyu-qwen3-235b-nvfp4",
                checkpoint_start_path=checkpoint_start,
                source_root=source_root,
        )


def test_cli_surface_is_fail_closed_and_complete() -> None:
    parser = gate.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {
        "prepare",
        "capture-checkpoint",
        "capture-provenance",
        "run",
        "assemble",
        "verify",
        "replay",
    }
    run_phase = next(action for action in choices["run"]._actions if action.dest == "phase")
    assert run_phase.choices == ("preflight", "formal")
    run_endpoint = next(action for action in choices["run"]._actions if action.dest == "endpoint")
    assert run_endpoint.choices == (gate.FORMAL_ENDPOINT,)
    checkpoint_destinations = {action.dest for action in choices["capture-checkpoint"]._actions}
    assert {"container", "source_root", "phase", "output"} <= checkpoint_destinations
    assert "model_path" not in checkpoint_destinations
    assert "model_volume" not in checkpoint_destinations
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--phase", "formal"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--phase",
                "formal",
                "--endpoint",
                "http://127.0.0.1:30001",
            ]
        )


def test_formal_artifact_contains_only_ten_required_generations(
    formal_artifact: dict[str, object],
) -> None:
    output_dir = formal_artifact["output_dir"]
    assert isinstance(output_dir, Path)
    rows = gate._read_jsonl(output_dir / gate.RAW_NAME)
    starts = [row for row in rows if row.get("type") == "scenario_start"]
    assert len(starts) == 10
    assert {row["phase"] for row in starts} == {"preflight", "formal"}
    assert not any(row.get("phase") in {"open-loop", "open_loop"} for row in rows)
