"""Focused CPU contracts for the replayable G4 M-A3 operator."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from bench import g2_a6_vllm_bench as a6
from bench import g4_ma3_sglang_bench as gate


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
        "fp4_gemm_backend": (
            gate.KAIRYU_FP4_GEMM_BACKEND
            if arm == "kairyu"
            else gate.SGLANG_FP4_GEMM_BACKEND
        ),
        "enable_alltoall": False,
        "moe_block_count": gate.NUM_LAYERS,
        "fallback_count": 0,
        "post_moe_allreduce": False,
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
                "bench/g4_ma3_kairyu_server.py",
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
            "source": gate.CHECKPOINT_ROOT,
            "destination": gate.CHECKPOINT_ROOT,
            "read_only": True,
        },
        "gpu_device_indices": list(gate.PHYSICAL_GPU_INDICES),
        "ipc_mode": "host",
        "shm_size_bytes": 32 * 1024**3,
        "cap_add": ["SYS_NICE", "SYS_PTRACE"],
    }
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "arm": arm,
        "server_generation_id": f"generation-{ordinal}",
        "server_started_ns": server_started_ns,
        "container": container,
        "source": source,
        "checkpoint": gate.expected_checkpoint(),
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
        "runtime": _runtime(arm, candidate),
    }


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
    return {
        "type": "request",
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
    provenance = gate.hashed_descriptor(
        _provenance(
            arm=arm,
            candidate=candidate,
            ordinal=ordinal,
            server_started_ns=server_started_ns,
        )
    )
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
    cursor = capture_start_ns + 1_000
    request_ordinal = 0
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
            "provenance_end": provenance,
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
    sweep_paths: list[Path] = []
    for arm in gate.ARMS:
        candidate = selection_value["selected"][arm]
        rows, end_ns = _scenario_rows(
            bundle=bundle,
            trace_bundle_sha256=trace_sha256,
            phase="open-loop",
            arm=arm,
            candidate=candidate,
            repeat=None,
            selection=selection,
            ordinal=ordinal,
            base_ns=base_ns,
            measurement_duration_ns=100_000,
            measurement_ttft_ns=10_000,
        )
        path = root / f"open-loop-{arm}.jsonl"
        gate.write_shard(path, rows)
        sweep_paths.append(path)
        base_ns = end_ns + 1_000
        ordinal += 1
    rows = gate.assemble_rows(
        preflight_paths=preflight_paths,
        formal_paths=formal_paths,
        sweep_paths=sweep_paths,
        selection_path=selection_path,
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
        "--schedule-policy fcfs --cuda-graph-backend-prefill disabled"
    ).split()
    assert gate.expected_sglang_argv(overlap_mode="default") == expected
    with pytest.raises(ValueError, match="pinned to 'none'"):
        gate.expected_sglang_argv(
            overlap_mode="default",
            moe_a2a_backend="flashinfer",
        )


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
        ("fp4_gemm_backend", "torch"),
        ("enable_alltoall", True),
        ("post_moe_allreduce", True),
    ):
        changed = copy.deepcopy(kairyu)
        changed[field] = invalid
        assert not gate._runtime_valid(
            changed,
            arm="kairyu",
            candidate="depth-5",
            moe_a2a_backend="none",
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


def test_preflight_winner_and_report_rates_are_raw_derived(
    formal_artifact: dict[str, object],
) -> None:
    selection = formal_artifact["selection"]
    assert selection["selected"] == {"kairyu": "depth-5", "sglang": "default"}
    assert selection["open_loop_rates_millirps"] == sorted(selection["open_loop_rates_millirps"])
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
        "scheduled-start",
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
    phase = "open-loop" if tamper == "scheduled-start" else "formal"
    start = next(
        row
        for row in rows
        if row.get("type") == "scenario_start"
        and row.get("phase") == phase
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
            "scheduled-start": "scheduled_start_ns",
        }[tamper]
        request = next(row for row in scenario_rows if field in row)
        request[field] = capture_start - 1
    with pytest.raises(gate.GateEvidenceError, match="capture interval"):
        gate.write_artifact(tmp_path / tamper, rows)


def test_open_loop_failure_remains_report_only(
    formal_artifact: dict[str, object], tmp_path: Path
) -> None:
    output_dir = formal_artifact["output_dir"]
    assert isinstance(output_dir, Path)
    rows = copy.deepcopy(gate._read_jsonl(output_dir / gate.RAW_NAME))
    request = next(
        row for row in rows if row.get("type") == "request" and row.get("phase") == "open_loop"
    )
    request["attempts"] = 2
    manifest = gate.write_artifact(tmp_path / "report-only", rows)
    assert manifest["passed"] is True
    assert (
        manifest["checks"]["report_only"]["open_loop_strict_streaming_successful_no_retry"] is False
    )


def test_cli_surface_is_fail_closed_and_complete() -> None:
    parser = gate.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"prepare", "run", "assemble", "verify", "replay"}
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--phase", "formal"])
