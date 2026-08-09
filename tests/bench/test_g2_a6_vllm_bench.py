"""Focused CPU contracts for the replayable G2 A6 comparison."""

from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path

import httpx
import pytest

from bench import g2_a6_vllm_bench as a6

_ENV_ARTIFACT_RELATIVE = "bench/results/env-2026-07-30.json"
_ENV_ARTIFACT_PATH = Path(a6.__file__).resolve().parents[1] / _ENV_ARTIFACT_RELATIVE
_ENV_ARTIFACT = json.loads(_ENV_ARTIFACT_PATH.read_text(encoding="utf-8"))
_ENV_MATRIX = _ENV_ARTIFACT["profile"]["p2p_matrix"]
_PCI_BUS_IDS = (
    "00000000:16:00.0",
    "00000000:27:00.0",
    "00000000:49:00.0",
    "00000000:5A:00.0",
    "00000000:98:00.0",
    "00000000:B8:00.0",
    "00000000:C8:00.0",
    "00000000:D8:00.0",
)


class _StableTokenizer:
    _piece = re.compile(r" p(\d{3})")

    def __init__(self, size: int = 256) -> None:
        self._pieces = [f" p{token_id:03d}" for token_id in range(size)]

    def vocab(self) -> list[str]:
        return list(self._pieces)

    def decode(self, token_ids) -> str:
        return "".join(self._pieces[token_id] for token_id in token_ids)

    def encode(self, text: str) -> tuple[int, ...]:
        result: list[int] = []
        position = 0
        for match in self._piece.finditer(text):
            if match.start() != position:
                raise ValueError("fixture text is not token aligned")
            result.append(int(match.group(1)))
            position = match.end()
        if position != len(text):
            raise ValueError("fixture text has an unmatched suffix")
        return tuple(result)


def _dataset_payload() -> str:
    records = [
        {
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        f" p{index % 200:03d}"
                        f" p{(index + 1) % 200:03d}"
                        f" p{(index + 2) % 200:03d}"
                        f" p{(index + 3) % 200:03d}"
                    ),
                },
                {"from": "gpt", "value": "fixture response"},
            ]
        }
        for index in range(160)
    ]
    return json.dumps(records)


@pytest.fixture(autouse=True)
def _pin_tiny_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Production pins the 673 MiB upstream file. CPU tests use a tiny
    # byte-pinned equivalent while exercising the same fail-closed schema.
    monkeypatch.setattr(
        a6,
        "SHAREGPT_DATASET_SHA256",
        a6.sha256_text(_dataset_payload()),
    )


def _dataset(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "sharegpt.json"
    path.write_text(_dataset_payload(), encoding="utf-8")
    digest = a6.sha256_file(path)
    return path, digest


def _configuration(tp: int, arm: str) -> dict[str, object]:
    return {
        "served_model_name": a6.MODEL,
        "dtype": a6.MODEL_DTYPE,
        "tensor_parallel_size": tp,
        "max_num_batched_tokens": a6.MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": a6.MAX_NUM_SEQS,
        "enable_prefix_caching": True,
        "cache_block_size": a6.CACHE_BLOCK_SIZE,
        "allocated_cache_blocks": (
            a6.KAIRYU_ALLOCATED_CACHE_BLOCKS
            if arm == "kairyu"
            else a6.VLLM_ALLOCATED_CACHE_BLOCKS
        ),
        "reserved_graph_scratch_blocks": (
            a6.KAIRYU_GRAPH_SCRATCH_BLOCKS
            if arm == "kairyu"
            else a6.VLLM_GRAPH_SCRATCH_BLOCKS
        ),
        "reserved_null_blocks": (
            a6.KAIRYU_NULL_BLOCKS
            if arm == "kairyu"
            else a6.VLLM_NULL_BLOCKS
        ),
        "usable_cache_blocks": a6.USABLE_CACHE_BLOCKS,
        "cache_tokens": a6.CACHE_TOKENS,
        "kv_cache_dtype": a6.KV_CACHE_DTYPE,
        "allocated_kv_cache_bytes_per_rank": (
            a6.ALLOCATED_KV_CACHE_BYTES_PER_RANK[tp]
        ),
        "reserved_kv_cache_bytes_per_rank": (
            a6.RESERVED_KV_CACHE_BYTES_PER_RANK[tp]
        ),
        "usable_kv_cache_bytes_per_rank": (
            a6.USABLE_KV_CACHE_BYTES_PER_RANK[tp]
        ),
        "chunked_prefill_enabled": True,
        "scheduler_policy": "fcfs",
        "access_log_enabled": False,
        "max_model_len": a6.MAX_MODEL_LEN,
        "pipeline_depth": a6.KAIRYU_PIPELINE_DEPTH if arm == "kairyu" else None,
        "async_scheduling": arm == "vllm-stock",
        "attention_backend": (
            "flashinfer" if arm == "kairyu" else a6.VLLM_ATTENTION_BACKEND
        ),
        "distributed_executor_backend": (
            "torch-distributed-nccl"
            if arm == "kairyu"
            else a6.VLLM_DISTRIBUTED_EXECUTOR_BACKEND
        ),
        "custom_all_reduce": False,
        "compilation": (
            None if arm == "kairyu" else a6.vllm_compilation_config()
        ),
        "cuda_graph": {
            "enabled": True,
            "mode": (
                "decode-only" if arm == "kairyu" else a6.VLLM_CUDAGRAPH_MODE
            ),
            "active_decode_graph_cap": a6.ACTIVE_DECODE_GRAPH_CAP,
            "capture_sizes": (
                list(a6.KAIRYU_CUDA_GRAPH_CAPTURE_SIZES)
                if arm == "kairyu"
                else list(a6.VLLM_CUDA_GRAPH_CAPTURE_SIZES)
            ),
        },
    }


def _configuration_source(tp: int, arm: str) -> dict[str, object] | None:
    if arm == "vllm-stock":
        return None
    template = (
        a6._REPO_ROOT
        / "bench/deploy/qwen3-32b-multi-gpu/g2-a6-kairyu.template.yaml"
    ).read_text(encoding="utf-8")
    rendered = template.replace("__TENSOR_PARALLEL_SIZE__", str(tp))
    return a6.kairyu_configuration_source(rendered, tp=tp)


def _vllm_startup_log(tp: int) -> dict[str, object]:
    config_message = (
        "Initializing a V1 LLM engine with config: "
        f"tensor_parallel_size={tp}, "
        f"max_seq_len={a6.MAX_MODEL_LEN}, "
        "enable_prefix_caching=True, "
        "enable_chunked_prefill=True, "
        "disable_custom_all_reduce=True, "
        f"VLLM_COMPILE {a6.VLLM_CUDAGRAPH_MODE} "
        f"{list(a6.VLLM_CUDA_GRAPH_CAPTURE_SIZES)}"
    )
    attention_messages = [a6.VLLM_ATTENTION_BACKEND_LOG_MARKER]
    attention_version_messages = [
        f"Using FlashAttention version {a6.VLLM_FLASH_ATTN_VERSION}"
    ]
    async_scheduling_messages = ["Asynchronous scheduling is enabled."]
    return a6.hashed_descriptor(
        {
            "engine_config_message": config_message,
            "engine_config_line_sha256": a6.sha256_text(config_message),
            "attention_backend_messages": attention_messages,
            "attention_backend_line_sha256": a6.sha256_json(
                attention_messages
            ),
            "attention_version_messages": attention_version_messages,
            "attention_version_line_sha256": a6.sha256_json(
                attention_version_messages
            ),
            "async_scheduling_messages": async_scheduling_messages,
            "async_scheduling_line_sha256": a6.sha256_json(
                async_scheduling_messages
            ),
            "resolved_markers": a6._vllm_startup_markers(
                config_message,
                attention_messages,
                attention_version_messages,
                async_scheduling_messages,
                tp=tp,
            ),
        }
    )


def _container_launch(
    *,
    tp: int,
    arm: str,
    generation: str,
    image_id: str,
    configuration_source: dict[str, object] | None,
) -> dict[str, object]:
    expected = a6.expected_container_launch(arm=arm, tp=tp)
    mounts: list[dict[str, object]] = [
        {
            "type": "volume",
            "name": "model-volume",
            "source": "volume:model-volume",
            "destination": "/models",
            "rw": False,
        },
        {
            "type": "bind",
            "name": None,
            "source": "repo:.",
            "destination": "/workspace",
            "rw": False,
        },
    ]
    engine_import: dict[str, object] = {
        "distribution": "kairyu" if arm == "kairyu" else "vllm",
        "package_file": (
            "/app/kairyu/__init__.py"
            if arm == "kairyu"
            else "/usr/local/lib/python3.12/site-packages/vllm/__init__.py"
        ),
        "package_sha256": a6.sha256_text(f"fixture-package-{arm}"),
    }
    if arm == "kairyu":
        assert isinstance(configuration_source, dict)
        source = configuration_source["value"]
        assert isinstance(source, dict)
        mounts.extend(
            (
                {
                    "type": "bind",
                    "name": None,
                    "source": "repo:kairyu",
                    "destination": "/app/kairyu",
                    "rw": False,
                },
                {
                    "type": "bind",
                    "name": None,
                    "source": f"generated-config:{source['text_sha256']}",
                    "destination": "/etc/kairyu/config.yaml",
                    "rw": False,
                },
            )
        )
        engine_import.update(
            {
                "engine_file": "/app/kairyu/engine/engine_loop.py",
                "engine_sha256": a6.sha256_text("fixture-engine"),
            }
        )
    return a6.hashed_descriptor(
        {
            "container_id": a6.sha256_text(f"container-{generation}"),
            "image_id": image_id,
            "config_image": image_id,
            "entrypoint": expected["entrypoint"],
            "argv": expected["argv"],
            "environment": expected["environment"],
            "working_dir": "/app" if arm == "kairyu" else "/root",
            "mounts": sorted(
                mounts,
                key=lambda item: str(item["destination"]),
            ),
            "host_config": {
                "ipc_mode": "host",
                "memlock_soft": -1,
                "memlock_hard": -1,
                "published_container_port": "8000/tcp",
                "published_host_ip": "127.0.0.1",
                "published_host_port": 18080,
                "gpu_device_ids": [str(index) for index in range(tp)],
            },
            "engine_import": engine_import,
        }
    )


def _provenance(
    *,
    tp: int,
    arm: str,
    generation: str,
    started_ns: int,
) -> dict[str, object]:
    configuration_source = _configuration_source(tp, arm)
    image_id = (
        f"sha256:{'2' * 64}"
        if arm == "kairyu"
        else a6.VLLM_IMAGE_ID
    )
    runtime = {
        "arm": arm,
        "version": "0.1.0" if arm == "kairyu" else a6.VLLM_VERSION,
        "build_commit": ("1" * 40 if arm == "kairyu" else a6.VLLM_BUILD_COMMIT),
        "image_ref": ("kairyu:test" if arm == "kairyu" else a6.VLLM_IMAGE_REF),
        "image_id": image_id,
        "stock_image": arm == "vllm-stock",
        "cuda_version": (
            a6.KAIRYU_CUDA_VERSION if arm == "kairyu" else a6.VLLM_CUDA_VERSION
        ),
        "nccl_version": (
            a6.KAIRYU_NCCL_VERSION if arm == "kairyu" else a6.VLLM_NCCL_VERSION
        ),
        "package_versions": {
            "torch": "fixture-torch",
            "flashinfer": "fixture-flashinfer",
            "uvloop": a6.UVLOOP_VERSION,
            "httptools": a6.HTTPTOOLS_VERSION,
            "uvicorn": (
                a6.KAIRYU_UVICORN_VERSION
                if arm == "kairyu"
                else a6.VLLM_UVICORN_VERSION
            ),
            ("kairyu" if arm == "kairyu" else "vllm"): (
                "0.1.0" if arm == "kairyu" else a6.VLLM_VERSION
            ),
        },
        "container_launch": _container_launch(
            tp=tp,
            arm=arm,
            generation=generation,
            image_id=image_id,
            configuration_source=configuration_source,
        ),
        "resolved_backend": a6.hashed_descriptor(
            a6.expected_resolved_backend(arm=arm, tp=tp)
        ),
    }
    if arm == "kairyu":
        runtime["source_clean"] = True
        runtime["source_status_policy"] = {
            "format": "porcelain-v1-z-untracked-files-all",
            "only_allowed_exception": f"?? {a6.SOURCE_CLEAN_EXCEPTION_PREFIX}*",
        }
    else:
        runtime["v1_engine"] = True
        runtime["startup_log"] = _vllm_startup_log(tp)
    peer_rates = [
        float(rate)
        for source, row in enumerate(_ENV_MATRIX)
        for destination, rate in enumerate(row)
        if source != destination
    ]
    return a6.hashed_descriptor(
        {
            "schema_version": 1,
            "server_generation": generation,
            "server_started_ns": started_ns,
            "host": {
                "id": "fixture-host",
                "environment": {
                    "measurement_session_id": _ENV_ARTIFACT[
                        "measurement_session_id"
                    ],
                    "measured_at_ns": _ENV_ARTIFACT["measured_at_ns"],
                    "artifact_date": _ENV_ARTIFACT["date"],
                    "artifact": {
                        "path": _ENV_ARTIFACT_RELATIVE,
                        "sha256": a6.sha256_file(_ENV_ARTIFACT_PATH),
                    },
                    "physical_topology_sha256": a6.sha256_text(_ENV_ARTIFACT["notes"]),
                    "fabric": {
                        "kind": _ENV_ARTIFACT["profile"]["interconnect"],
                        "raw_p2p_matrix_sha256": a6.sha256_json(_ENV_MATRIX),
                        "minimum_peer_bandwidth_gbs": min(peer_rates),
                    },
                },
            },
            "gpu": {
                "uuids": [f"GPU-{index}" for index in range(tp)],
                "pci_bus_ids": list(_PCI_BUS_IDS[:tp]),
                "product_name": _ENV_ARTIFACT["profile"]["device_name"],
                "driver_version": _ENV_ARTIFACT["driver"],
                "container_visible_uuids": [
                    f"GPU-{index}" for index in range(tp)
                ],
                "container_visible_pci_bus_ids": list(_PCI_BUS_IDS[:tp]),
            },
            "model": a6.model_identity(),
            "checkpoint": a6.hashed_descriptor(
                {
                    "schema_version": 1,
                    "model": a6.MODEL,
                    "model_revision": a6.MODEL_REVISION,
                    "checkpoint_root": a6.CHECKPOINT_ROOT,
                    "checkpoint": a6._expected_checkpoint_identity(),
                    "revision_files": [
                        {
                            "file": (
                                ".cache/huggingface/download/"
                                "model.safetensors.index.json.metadata"
                            ),
                            "revision": a6.MODEL_REVISION,
                        }
                    ],
                }
            ),
            "runtime": runtime,
            "configuration": _configuration(tp, arm),
            "configuration_source": configuration_source,
        }
    )


def _request_row(
    *,
    scenario: tuple[int, str, int, str],
    expected: dict[str, object],
    phase: str,
    ttft_ns: int,
    total_ns: int,
) -> dict[str, object]:
    tp, workload, repeat, arm = scenario
    sequence = int(expected["sequence"])
    base = (
        tp * 10**15
        + a6.WORKLOADS.index(workload) * 10**14
        + repeat * 10**13
        + a6.ARMS.index(arm) * 10**12
    )
    if phase == "warmup":
        start = base - (a6.WARMUP_REQUESTS - sequence) * 1_000_000
    elif phase == "graph_warmup":
        offset = 0
        batch_index = 0
        batch_size = 0
        batch_sequence = 0
        for index, candidate in enumerate(a6.GRAPH_WARMUP_BATCH_SIZES):
            if sequence < offset + candidate:
                batch_index = index
                batch_size = candidate
                batch_sequence = sequence - offset
                break
            offset += candidate
        start = base - 500_000 + batch_index * 10_000
    else:
        start = base if workload == "sharegpt" else base + sequence * 1_000_000
    completion_tokens = (
        a6.WARMUP_MAX_TOKENS
        if phase == "warmup"
        else (
            a6.GRAPH_WARMUP_COMPLETION_TOKENS
            if phase == "graph_warmup"
            else int(a6.request_config(workload)["max_tokens"])
        )
    )
    return {
        "type": "request",
        "scenario_id": a6._scenario_id(*scenario),
        "tensor_parallel_size": tp,
        "workload": workload,
        "repeat": repeat,
        "arm": arm,
        "phase": phase,
        "sequence": sequence,
        "attempts": 1,
        "prompt_text_sha256": expected["prompt_text_sha256"],
        "prompt_tokens": expected["prompt_tokens"],
        "prompt_token_ids_sha256": expected["prompt_token_ids_sha256"],
        "expected_completion_tokens": completion_tokens,
        "http_status": 200,
        "content_type": "text/event-stream; charset=utf-8",
        "response_request_header": f"header-{sequence}",
        "response_id": f"response-{tp}-{workload}-{repeat}-{arm}-{phase}-{sequence}",
        "status": "success",
        "error_type": None,
        "done": True,
        "done_events": 1,
        "terminal_choice": True,
        "terminal_choice_events": 1,
        "finish_reasons": ["length"],
        "usage_events": 1,
        "usage": {
            "prompt_tokens": expected["prompt_tokens"],
            "completion_tokens": completion_tokens,
            "cached_tokens": 0,
        },
        "timing_ns": {
            "start": start,
            "first_token": start + ttft_ns,
            "end": start + total_ns,
            "ttft": ttft_ns,
            "total": total_ns,
        },
        **(
            {"burst_release_ns": base - 1}
            if workload == "sharegpt" and phase == "measurement"
            else {}
        ),
        **(
            {
                "graph_batch_size": batch_size,
                "graph_batch_sequence": batch_sequence,
                "graph_warmup_release_ns": start - 1,
            }
            if phase == "graph_warmup"
            else {}
        ),
        "output_text_sha256": "3" * 64,
        "stream_payload_sha256": "4" * 64,
    }


def _passing_rows(tmp_path: Path) -> list[dict[str, object]]:
    dataset, dataset_sha256 = _dataset(tmp_path)
    tokenizer = _StableTokenizer()
    traces = {
        "sharegpt": a6.build_sharegpt_trace(
            dataset,
            tokenizer,
            expected_dataset_sha256=dataset_sha256,
        ),
        "shared-prefix": a6.build_shared_prefix_trace(tokenizer),
    }
    graph_warmup = a6.build_graph_warmup_trace(tokenizer)
    assert a6._graph_warmup_disjoint(graph_warmup, traces)
    rows: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": a6.SCHEMA_VERSION,
            "gate": a6.GATE,
            "benchmark": a6.benchmark_config(dataset_sha256),
            "graph_warmup": graph_warmup,
            "traces": traces,
        }
    ]
    generation = 0
    for scenario in a6.EXPECTED_SCENARIOS:
        tp, workload, repeat, arm = scenario
        first_arm, second_arm = a6.PAIR_ORDER[repeat]
        pair_base = tp * 1_000_000 + a6.WORKLOADS.index(workload) * 100_000 + repeat * 10
        started_ns = pair_base + (0 if arm == first_arm else 1 if arm == second_arm else 2)
        provenance = _provenance(
            tp=tp,
            arm=arm,
            generation=f"generation-{generation}",
            started_ns=started_ns,
        )
        generation += 1
        scenario_id = a6._scenario_id(*scenario)
        trace = traces[workload]
        rows.append(
            {
                "type": "scenario_start",
                "schema_version": a6.SCHEMA_VERSION,
                "gate": a6.GATE,
                "scenario_id": scenario_id,
                "tensor_parallel_size": tp,
                "workload": workload,
                "repeat": repeat,
                "arm": arm,
                "endpoint": "http://fixture",
                "benchmark": a6.benchmark_config(dataset_sha256),
                "trace_sha256": trace["sha256"],
                "graph_warmup_sha256": graph_warmup["sha256"],
                "measurement_raw_sha256": "5" * 64,
                "provenance_start": provenance,
            }
        )
        raw_trace = trace["value"]
        assert isinstance(raw_trace, dict)
        warmup_rows = []
        for expected in raw_trace["warmup"]:
            warmup_rows.append(
                _request_row(
                    scenario=scenario,
                    expected=expected,
                    phase="warmup",
                    ttft_ns=10,
                    total_ns=20,
                )
            )
        graph_warmup_rows = []
        for expected in graph_warmup["value"]["prompts"]:
            graph_warmup_rows.append(
                _request_row(
                    scenario=scenario,
                    expected=expected,
                    phase="graph_warmup",
                    ttft_ns=10,
                    total_ns=20,
                )
            )
        formal_rows = []
        if arm == "kairyu":
            ttft_ns = 80
            total_ns = 200 if workload == "sharegpt" else 100
        else:
            ttft_ns = 100
            total_ns = 190 if workload == "sharegpt" else 120
        for expected in raw_trace["requests"]:
            formal_rows.append(
                _request_row(
                    scenario=scenario,
                    expected=expected,
                    phase="measurement",
                    ttft_ns=ttft_ns,
                    total_ns=total_ns,
                )
            )
        rows.extend((*warmup_rows, *graph_warmup_rows, *formal_rows))
        rows.append(
            {
                "type": "scenario_end",
                "scenario_id": scenario_id,
                "tensor_parallel_size": tp,
                "workload": workload,
                "repeat": repeat,
                "arm": arm,
                "warmup_request_count": len(warmup_rows)
                + len(graph_warmup_rows),
                "serial_warmup_request_count": len(warmup_rows),
                "graph_warmup_request_count": len(graph_warmup_rows),
                "measurement_request_count": len(formal_rows),
                "measurement_window_ns": {
                    "start": min(row["timing_ns"]["start"] for row in formal_rows),
                    "end": max(row["timing_ns"]["end"] for row in formal_rows),
                },
                "provenance_end": provenance,
            }
        )
    share_trace = traces["sharegpt"]
    share_raw = share_trace["value"]
    assert isinstance(share_raw, dict)
    for sweep_scenario in a6.EXPECTED_SWEEP_SCENARIOS:
        tp, rate_rps, arm = sweep_scenario
        scenario_id = a6._sweep_scenario_id(*sweep_scenario)
        provenance = _provenance(
            tp=tp,
            arm=arm,
            generation=f"generation-{generation}",
            started_ns=10_000_000 + generation,
        )
        generation += 1
        rows.append(
            {
                "type": "sweep_scenario_start",
                "schema_version": a6.SCHEMA_VERSION,
                "gate": a6.GATE,
                "scenario_id": scenario_id,
                "tensor_parallel_size": tp,
                "arm": arm,
                "arrival_rate_rps": rate_rps,
                "endpoint": "http://fixture",
                "benchmark": a6.benchmark_config(dataset_sha256),
                "trace_sha256": share_trace["sha256"],
                "graph_warmup_sha256": graph_warmup["sha256"],
                "measurement_raw_sha256": "6" * 64,
                "provenance_start": provenance,
            }
        )
        warmup_rows = []
        binding_scenario = (tp, "sharegpt", 0, arm)
        for expected in share_raw["warmup"]:
            row = _request_row(
                scenario=binding_scenario,
                expected=expected,
                phase="warmup",
                ttft_ns=80,
                total_ns=100,
            )
            warmup_rows.append(
                {
                    **row,
                    "type": "sweep_request",
                    "scenario_id": scenario_id,
                    "arrival_rate_rps": rate_rps,
                }
            )
        graph_warmup_rows = []
        for expected in graph_warmup["value"]["prompts"]:
            row = _request_row(
                scenario=binding_scenario,
                expected=expected,
                phase="graph_warmup",
                ttft_ns=10,
                total_ns=20,
            )
            graph_warmup_rows.append(
                {
                    **row,
                    "type": "sweep_request",
                    "scenario_id": scenario_id,
                    "arrival_rate_rps": rate_rps,
                }
            )
        sweep_rows = []
        anchor_ns = tp * 10**15 + rate_rps * 10**12 + a6.ARMS.index(arm) * 10**11
        period_ns = 1_000_000_000 // rate_rps
        for expected in share_raw["requests"]:
            sequence = int(expected["sequence"])
            row = _request_row(
                scenario=binding_scenario,
                expected=expected,
                phase="sweep",
                ttft_ns=80 if arm == "kairyu" else 100,
                total_ns=200,
            )
            scheduled_start = anchor_ns + sequence * period_ns
            row["timing_ns"] = {
                "start": scheduled_start,
                "first_token": scheduled_start + (80 if arm == "kairyu" else 100),
                "end": scheduled_start + 200,
                "ttft": 80 if arm == "kairyu" else 100,
                "total": 200,
            }
            sweep_rows.append(
                {
                    **row,
                    "type": "sweep_request",
                    "scenario_id": scenario_id,
                    "arrival_rate_rps": rate_rps,
                    "scheduled_offset_ns": sequence * period_ns,
                    "scheduled_start_ns": scheduled_start,
                    "release_lag_ns": 0,
                }
            )
        rows.extend((*warmup_rows, *graph_warmup_rows, *sweep_rows))
        rows.append(
            {
                "type": "sweep_scenario_end",
                "scenario_id": scenario_id,
                "tensor_parallel_size": tp,
                "arm": arm,
                "arrival_rate_rps": rate_rps,
                "warmup_request_count": len(warmup_rows)
                + len(graph_warmup_rows),
                "serial_warmup_request_count": len(warmup_rows),
                "graph_warmup_request_count": len(graph_warmup_rows),
                "sweep_request_count": len(sweep_rows),
                "measurement_window_ns": {
                    "start": min(row["timing_ns"]["start"] for row in sweep_rows),
                    "end": max(row["timing_ns"]["end"] for row in sweep_rows),
                },
                "arrival_anchor_ns": anchor_ns,
                "provenance_end": provenance,
            }
        )
    binding_request_count = len(a6.EXPECTED_SCENARIOS) * (
        a6.WARMUP_REQUESTS + a6.GRAPH_WARMUP_REQUESTS
    ) + len(a6.TP_DEGREES) * len(a6.ARMS) * a6.REPEATS * (
        a6.SHAREGPT_REQUESTS + a6.SHARED_PREFIX_REQUESTS
    )
    sweep_request_count = len(a6.EXPECTED_SWEEP_SCENARIOS) * (
        a6.WARMUP_REQUESTS
        + a6.GRAPH_WARMUP_REQUESTS
        + a6.SHAREGPT_REQUESTS
    )
    rows.append(
        {
            "type": "run_end",
            "scenario_count": len(a6.EXPECTED_SCENARIOS),
            "sweep_scenario_count": len(a6.EXPECTED_SWEEP_SCENARIOS),
            "binding_request_count": binding_request_count,
            "sweep_request_count": sweep_request_count,
            "request_count": binding_request_count + sweep_request_count,
        }
    )
    return rows


def test_fixed_text_workloads_retain_arm_neutral_token_ids_and_hashes(
    tmp_path: Path,
) -> None:
    dataset, digest = _dataset(tmp_path)
    tokenizer = _StableTokenizer()
    sharegpt = a6.build_sharegpt_trace(
        dataset,
        tokenizer,
        expected_dataset_sha256=digest,
    )
    shared = a6.build_shared_prefix_trace(tokenizer)
    graph_warmup = a6.build_graph_warmup_trace(tokenizer)

    assert a6._trace_valid(
        sharegpt,
        workload="sharegpt",
        dataset_sha256=digest,
    )
    assert a6._trace_valid(
        shared,
        workload="shared-prefix",
        dataset_sha256=digest,
    )
    share_rows = sharegpt["value"]["requests"]
    assert len(share_rows) == 128
    assert len({row["source_index"] for row in share_rows}) == 128
    expected_order = list(range(160))
    random.Random(0).shuffle(expected_order)
    assert [row["source_index"] for row in share_rows] == expected_order[:128]
    assert [row["source_index"] for row in sharegpt["value"]["warmup"]] == expected_order[128:132]
    for row in share_rows:
        assert tokenizer.encode(row["prompt_text"]) == tuple(row["prompt_token_ids"])
        assert row["prompt_text_sha256"] == a6.sha256_text(row["prompt_text"])
    shared_rows = shared["value"]["requests"]
    assert len(shared_rows) == 512
    assert sum(row["prompt_tokens"] for row in shared_rows) == 557_056
    for row in shared_rows:
        assert tokenizer.encode(row["prompt_text"]) == tuple(row["prompt_token_ids"])
    bundle = {
        "schema_version": a6.TRACE_BUNDLE_SCHEMA,
        "benchmark": a6.benchmark_config(digest),
        "graph_warmup": graph_warmup,
        "traces": {
            "sharegpt": sharegpt,
            "shared-prefix": shared,
        },
    }
    graph_prompts = graph_warmup["value"]["prompts"]
    assert len(graph_prompts) == 31
    assert len(
        {row["prompt_token_ids_sha256"] for row in graph_prompts}
    ) == 31
    assert all(row["prompt_tokens"] == 64 for row in graph_prompts)
    assert a6._graph_warmup_disjoint(graph_warmup, bundle["traces"])
    duplicate_graph = copy.deepcopy(graph_warmup["value"])
    duplicate_graph["prompts"][1] = copy.deepcopy(
        duplicate_graph["prompts"][0]
    )
    duplicate_graph["prompts"][1]["sequence"] = 1
    assert not a6._graph_warmup_trace_valid(
        a6.hashed_descriptor(duplicate_graph)
    )
    bundle_path = tmp_path / "traces.json"
    bundle_path.write_text(a6.canonical_json(bundle) + "\n", encoding="utf-8")
    assert (
        a6.load_trace_bundle(
            bundle_path,
            dataset_sha256=digest,
        )
        == bundle
    )

    undersized_raw = copy.deepcopy(sharegpt["value"])
    undersized_row = undersized_raw["requests"][0]
    undersized_ids = undersized_row["prompt_token_ids"][:3]
    undersized_text = tokenizer.decode(undersized_ids)
    undersized_row.update(
        {
            "prompt_text": undersized_text,
            "prompt_text_sha256": a6.sha256_text(undersized_text),
            "prompt_token_ids": undersized_ids,
            "prompt_tokens": len(undersized_ids),
            "prompt_token_ids_sha256": a6.sha256_json(undersized_ids),
        }
    )
    assert not a6._trace_valid(
        a6.hashed_descriptor(undersized_raw),
        workload="sharegpt",
        dataset_sha256=digest,
    )


@pytest.mark.asyncio
async def test_strict_sse_error_is_retained_without_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = 'data: {"error":{"message":"boom"}}\n\ndata: [DONE]\n\n'
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    planned = a6.PlannedRequest(
        phase="measurement",
        sequence=0,
        prompt_text=" p001",
        prompt_token_ids=(1,),
        expected_completion_tokens=a6.SHAREGPT_MAX_TOKENS,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await a6.stream_completion_once(
            client,
            endpoint="http://fixture",
            planned=planned,
            workload="sharegpt",
        )

    assert calls == 1
    assert row["attempts"] == 1
    assert row["status"] == "failure"
    assert row["error_type"] == "SSEError"
    assert row["done"] is True


@pytest.mark.asyncio
async def test_formal_request_sends_text_and_fixed_length_sampling_once() -> None:
    calls = 0
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        captured.update(json.loads(request.content))
        body = "\n\n".join(
            (
                (
                    'data: {"id":"cmpl-1","choices":'
                    '[{"index":0,"text":"x","finish_reason":null}],'
                    '"usage":null}'
                ),
                (
                    'data: {"id":"cmpl-1","choices":'
                    '[{"index":0,"text":"","finish_reason":"length"}],'
                    '"usage":null}'
                ),
                (
                    'data: {"id":"cmpl-1","choices":[],"usage":'
                    '{"prompt_tokens":2,"completion_tokens":128,'
                    '"total_tokens":130}}'
                ),
                "data: [DONE]",
                "",
            )
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    planned = a6.PlannedRequest(
        phase="measurement",
        sequence=0,
        prompt_text=" p001 p002",
        prompt_token_ids=(1, 2),
        expected_completion_tokens=a6.SHAREGPT_MAX_TOKENS,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await a6.stream_completion_once(
            client,
            endpoint="http://fixture",
            planned=planned,
            workload="sharegpt",
        )

    assert calls == 1
    assert captured == {
        "model": a6.MODEL,
        "prompt": " p001 p002",
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 0,
        "n": 1,
        "max_tokens": 128,
        "min_tokens": 128,
        "ignore_eos": True,
    }
    assert row["status"] == "success"
    assert row["attempts"] == 1
    assert row["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 128,
        "cached_tokens": 0,
    }
    assert row["prompt_text_sha256"] == a6.sha256_text(" p001 p002")
    assert row["prompt_token_ids_sha256"] == a6.sha256_json([1, 2])


@pytest.mark.asyncio
async def test_empty_decoding_first_token_is_still_the_ttft_event() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        body = "\n\n".join(
            (
                (
                    'data: {"id":"cmpl-empty","choices":'
                    '[{"index":0,"text":"","finish_reason":"length"}],'
                    '"usage":null}'
                ),
                (
                    'data: {"id":"cmpl-empty","choices":[],"usage":'
                    '{"prompt_tokens":1,"completion_tokens":1,'
                    '"total_tokens":2}}'
                ),
                "data: [DONE]",
                "",
            )
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    planned = a6.PlannedRequest(
        phase="measurement",
        sequence=0,
        prompt_text=" p001",
        prompt_token_ids=(1,),
        expected_completion_tokens=1,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await a6.stream_completion_once(
            client,
            endpoint="http://fixture",
            planned=planned,
            workload="shared-prefix",
        )

    assert row["status"] == "success"
    assert row["timing_ns"]["first_token"] is not None
    assert row["timing_ns"]["ttft"] >= 0
    assert row["output_text_sha256"] == a6.sha256_text("")


@pytest.mark.asyncio
async def test_non_length_terminal_is_rejected_for_fixed_length_workload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        body = "\n\n".join(
            (
                (
                    'data: {"id":"cmpl-stop","choices":'
                    '[{"index":0,"text":"x","finish_reason":"stop"}],'
                    '"usage":null}'
                ),
                (
                    'data: {"id":"cmpl-stop","choices":[],"usage":'
                    '{"prompt_tokens":1,"completion_tokens":1,'
                    '"total_tokens":2}}'
                ),
                "data: [DONE]",
                "",
            )
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    planned = a6.PlannedRequest(
        phase="measurement",
        sequence=0,
        prompt_text=" p001",
        prompt_token_ids=(1,),
        expected_completion_tokens=1,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await a6.stream_completion_once(
            client,
            endpoint="http://fixture",
            planned=planned,
            workload="shared-prefix",
        )

    assert row["status"] == "failure"
    assert row["error_type"] == "UnexpectedFinishReason"
    assert row["finish_reasons"] == ["stop"]


@pytest.mark.asyncio
async def test_serial_executor_continues_after_a_failed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[int] = []

    async def fake_once(client, *, endpoint, planned, workload):
        del client, endpoint, workload
        attempted.append(planned.sequence)
        return {
            "sequence": planned.sequence,
            "status": "failure" if planned.sequence == 1 else "success",
        }

    monkeypatch.setattr(a6, "stream_completion_once", fake_once)
    planned = [
        a6.PlannedRequest(
            phase="measurement",
            sequence=sequence,
            prompt_text=f" p{sequence:03d}",
            prompt_token_ids=(sequence,),
            expected_completion_tokens=1,
        )
        for sequence in range(3)
    ]
    rows = await a6._execute_serial(
        object(),
        endpoint="http://fixture",
        planned=planned,
        workload="shared-prefix",
    )

    assert attempted == [0, 1, 2]
    assert [row["status"] for row in rows] == [
        "success",
        "failure",
        "success",
    ]


def test_nearest_rank_has_exact_boundary_semantics() -> None:
    assert a6.nearest_rank(range(1, 101), 0.50) == 50
    assert a6.nearest_rank(range(1, 101), 0.99) == 99
    assert a6.nearest_rank(range(1, 102), 0.99) == 100
    with pytest.raises(ValueError, match="requires samples"):
        a6.nearest_rank([], 0.99)


def test_complete_matrix_passes_inclusive_unrounded_threshold_boundaries(
    tmp_path: Path,
) -> None:
    rows = _passing_rows(tmp_path)
    checks, comparisons, diagnostics = a6.evaluate_rows(
        [{**row, "row_index": index} for index, row in enumerate(rows)]
    )

    assert all(checks.values())
    for tp in a6.TP_DEGREES:
        result = comparisons[f"tp{tp}"]
        assert result["sharegpt"]["kairyu_over_vllm_goodput_ratio"] == pytest.approx(0.95)
        assert result["sharegpt"]["kairyu_over_vllm_ttft_p99_ratio"] == pytest.approx(0.8)
        assert result["shared-prefix"]["kairyu_over_vllm_ttft_p50_ratio"] == pytest.approx(0.8)
        assert len(result["paired_rounds"]) == a6.REPEATS
        assert result["paired_median"]["binding"] is False
        for pair in result["paired_rounds"]:
            for key in (
                "sharegpt_goodput_kairyu_over_vllm",
                "sharegpt_ttft_p99_kairyu_over_vllm",
                "shared_prefix_ttft_p50_kairyu_over_vllm",
            ):
                assert set(pair[key]) == {
                    "numerator",
                    "denominator",
                    "decimal",
                }
    assert diagnostics["method"] == {
        "warmup_in_raw_but_excluded": True,
        "outlier_removal": False,
        "gate_uses_unrounded_values": True,
        "paired_order_diagnostics": (
            "four balanced K/V, V/K, V/K, K/V repeats reduce order bias; "
            "exact-Fraction per-repeat ratios and median/MAD disclose "
            "variability and are non-binding"
        ),
        "open_loop_arrival_sweep_is_binding": False,
    }


def test_goodput_gate_uses_exact_integer_cross_multiplication(
    tmp_path: Path,
) -> None:
    rows = _passing_rows(tmp_path)
    for row in rows:
        if (
            row.get("type") == "request"
            and row.get("phase") == "measurement"
            and row.get("workload") == "sharegpt"
        ):
            duration = 740 if row["arm"] == "kairyu" else 703
            row["timing_ns"]["end"] = row["timing_ns"]["start"] + duration
            row["timing_ns"]["total"] = duration
    for row in rows:
        if row.get("type") == "scenario_end" and row.get("workload") == "sharegpt":
            formal = [
                candidate
                for candidate in rows
                if candidate.get("type") == "request"
                and candidate.get("phase") == "measurement"
                and candidate.get("scenario_id") == row["scenario_id"]
            ]
            row["measurement_window_ns"] = {
                "start": min(candidate["timing_ns"]["start"] for candidate in formal),
                "end": max(candidate["timing_ns"]["end"] for candidate in formal),
            }

    checks, comparisons, _ = a6.evaluate_rows(
        [{**row, "row_index": index} for index, row in enumerate(rows)]
    )

    assert 703 * 100 == 740 * 95
    assert all(checks.values())
    for tp in a6.TP_DEGREES:
        assert comparisons[f"tp{tp}"]["sharegpt"]["kairyu_over_vllm_goodput_ratio"] < 0.95


@pytest.mark.parametrize(
    "mutation,failed_check",
    [
        (
            "missing_scenario",
            "required_tp_arm_workload_repeat_matrix_exact",
        ),
        (
            "wrong_pair_order",
            "balanced_kv_vk_vk_kv_server_start_order",
        ),
        (
            "reused_generation",
            "fresh_server_generation_per_scenario",
        ),
        (
            "different_kairyu_build_commit",
            "runtime_and_full_config_stable_within_arm_tp",
        ),
        (
            "mismatched_gpu",
            "gpu_identity_exact_with_tp4_subset_of_tp8",
        ),
        (
            "different_measurement_session",
            "model_and_host_identity_exact",
        ),
        (
            "prompt_identity",
            "arm_neutral_trace_identity_and_geometry_exact",
        ),
        (
            "request_failure",
            "strict_streaming_all_requests_successful_without_retry",
        ),
        (
            "serialized_sharegpt",
            "sharegpt_synchronized_concurrency_128_shape_exact",
        ),
        (
            "invalid_burst_release",
            "sharegpt_synchronized_concurrency_128_shape_exact",
        ),
        (
            "parallel_shared_prefix",
            "shared_prefix_serial_execution_shape_exact",
        ),
        (
            "graph_warmup_release",
            "fixed_synchronized_graph_warmup_precedes_measurement_exact",
        ),
        (
            "graph_warmup_prompt",
            "fixed_synchronized_graph_warmup_precedes_measurement_exact",
        ),
        (
            "sweep_schedule",
            "open_loop_fixed_arrival_schedule_replays_exact",
        ),
        (
            "missing_sweep_point",
            "report_only_open_loop_rate_grid_exact",
        ),
    ],
)
def test_matrix_provenance_prompt_and_failure_mutations_are_rejected(
    tmp_path: Path,
    mutation: str,
    failed_check: str,
) -> None:
    rows = _passing_rows(tmp_path)
    if mutation == "missing_scenario":
        scenario_id = a6._scenario_id(4, "sharegpt", 0, "kairyu")
        rows = [row for row in rows if row.get("scenario_id") != scenario_id]
    elif mutation == "wrong_pair_order":
        start = next(
            row
            for row in rows
            if row.get("type") == "scenario_start"
            and row.get("scenario_id") == a6._scenario_id(4, "sharegpt", 0, "kairyu")
        )
        value = copy.deepcopy(start["provenance_start"]["value"])
        value["server_started_ns"] += 2
        start["provenance_start"] = a6.hashed_descriptor(value)
        end = next(
            row
            for row in rows
            if row.get("type") == "scenario_end" and row.get("scenario_id") == start["scenario_id"]
        )
        end["provenance_end"] = start["provenance_start"]
    elif mutation == "reused_generation":
        starts = [row for row in rows if row.get("type") == "scenario_start"]
        value = copy.deepcopy(starts[1]["provenance_start"]["value"])
        value["server_generation"] = starts[0]["provenance_start"]["value"]["server_generation"]
        starts[1]["provenance_start"] = a6.hashed_descriptor(value)
        end = next(
            row
            for row in rows
            if row.get("type") == "scenario_end"
            and row.get("scenario_id") == starts[1]["scenario_id"]
        )
        end["provenance_end"] = starts[1]["provenance_start"]
    elif mutation == "different_kairyu_build_commit":
        start = next(
            row
            for row in rows
            if row.get("type") == "scenario_start"
            and row.get("scenario_id") == a6._scenario_id(4, "sharegpt", 0, "kairyu")
        )
        value = copy.deepcopy(start["provenance_start"]["value"])
        value["runtime"]["build_commit"] = "a" * 40
        start["provenance_start"] = a6.hashed_descriptor(value)
        end = next(
            row
            for row in rows
            if row.get("type") == "scenario_end" and row.get("scenario_id") == start["scenario_id"]
        )
        end["provenance_end"] = start["provenance_start"]
    elif mutation == "mismatched_gpu":
        start = next(
            row
            for row in rows
            if row.get("type") == "scenario_start"
            and row.get("scenario_id") == a6._scenario_id(4, "sharegpt", 0, "vllm-stock")
        )
        value = copy.deepcopy(start["provenance_start"]["value"])
        value["gpu"]["uuids"][0] = "GPU-other"
        start["provenance_start"] = a6.hashed_descriptor(value)
        end = next(
            row
            for row in rows
            if row.get("type") == "scenario_end" and row.get("scenario_id") == start["scenario_id"]
        )
        end["provenance_end"] = start["provenance_start"]
    elif mutation == "different_measurement_session":
        start = next(
            row
            for row in rows
            if row.get("type") == "scenario_start"
            and row.get("scenario_id") == a6._scenario_id(4, "sharegpt", 0, "vllm-stock")
        )
        value = copy.deepcopy(start["provenance_start"]["value"])
        value["host"]["environment"]["measurement_session_id"] = "other-session"
        start["provenance_start"] = a6.hashed_descriptor(value)
        end = next(
            row
            for row in rows
            if row.get("type") == "scenario_end" and row.get("scenario_id") == start["scenario_id"]
        )
        end["provenance_end"] = start["provenance_start"]
    elif mutation == "prompt_identity":
        row = next(
            row
            for row in rows
            if row.get("type") == "request" and row.get("phase") == "measurement"
        )
        row["prompt_token_ids_sha256"] = "9" * 64
    elif mutation == "request_failure":
        row = next(
            row
            for row in rows
            if row.get("type") == "request" and row.get("phase") == "measurement"
        )
        row["status"] = "failure"
        row["error_type"] = "HTTPError"
    elif mutation == "serialized_sharegpt":
        scenario_id = a6._scenario_id(4, "sharegpt", 0, "kairyu")
        formal = [
            row
            for row in rows
            if row.get("type") == "request"
            and row.get("scenario_id") == scenario_id
            and row.get("phase") == "measurement"
        ]
        base = min(row["timing_ns"]["start"] for row in formal)
        for sequence, row in enumerate(formal):
            duration = row["timing_ns"]["total"]
            ttft = row["timing_ns"]["ttft"]
            start = base + sequence * (duration + 1)
            row["timing_ns"].update(
                {
                    "start": start,
                    "first_token": start + ttft,
                    "end": start + duration,
                }
            )
        end = next(
            row
            for row in rows
            if row.get("type") == "scenario_end" and row.get("scenario_id") == scenario_id
        )
        end["measurement_window_ns"] = {
            "start": min(row["timing_ns"]["start"] for row in formal),
            "end": max(row["timing_ns"]["end"] for row in formal),
        }
    elif mutation == "invalid_burst_release":
        row = next(
            row
            for row in rows
            if row.get("type") == "request"
            and row.get("workload") == "sharegpt"
            and row.get("phase") == "measurement"
        )
        row["burst_release_ns"] = row["timing_ns"]["start"] + 1
    elif mutation == "parallel_shared_prefix":
        scenario_id = a6._scenario_id(4, "shared-prefix", 0, "kairyu")
        formal = [
            row
            for row in rows
            if row.get("type") == "request"
            and row.get("scenario_id") == scenario_id
            and row.get("phase") == "measurement"
        ]
        start = min(row["timing_ns"]["start"] for row in formal)
        for row in formal:
            duration = row["timing_ns"]["total"]
            ttft = row["timing_ns"]["ttft"]
            row["timing_ns"].update(
                {
                    "start": start,
                    "first_token": start + ttft,
                    "end": start + duration,
                }
            )
        end = next(
            row
            for row in rows
            if row.get("type") == "scenario_end" and row.get("scenario_id") == scenario_id
        )
        end["measurement_window_ns"] = {
            "start": start,
            "end": max(row["timing_ns"]["end"] for row in formal),
        }
    elif mutation == "graph_warmup_release":
        row = next(
            row
            for row in rows
            if row.get("type") == "request"
            and row.get("phase") == "graph_warmup"
            and row.get("graph_batch_size") == 16
        )
        row["graph_warmup_release_ns"] = row["timing_ns"]["start"] + 1
    elif mutation == "graph_warmup_prompt":
        row = next(
            row
            for row in rows
            if row.get("type") == "request"
            and row.get("phase") == "graph_warmup"
        )
        row["prompt_token_ids_sha256"] = "9" * 64
    elif mutation == "sweep_schedule":
        row = next(
            row
            for row in rows
            if row.get("type") == "sweep_request" and row.get("phase") == "sweep"
        )
        row["scheduled_offset_ns"] += 1
    else:
        scenario_id = a6._sweep_scenario_id(4, 1, "kairyu")
        rows = [row for row in rows if row.get("scenario_id") != scenario_id]

    normalized = [{**row, "row_index": index} for index, row in enumerate(rows)]
    checks, _, _ = a6.evaluate_rows(normalized)
    assert checks[failed_check] is False


def test_raw_and_manifest_tampering_are_independently_rejected(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    rows = _passing_rows(tmp_path)
    manifest = a6.write_artifact(artifact, rows)
    assert manifest["passed"] is True
    assert a6.verify_artifact(artifact, assert_gate=True) == manifest

    manifest_path = artifact / a6.MANIFEST_NAME
    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["passed"] = False
    manifest_path.write_text(
        json.dumps(tampered_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="independent raw replay"):
        a6.verify_artifact(artifact)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    raw_path = artifact / a6.RAW_NAME
    raw_path.write_text(
        raw_path.read_text().replace('"ttft":80', '"ttft":81', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        a6.verify_artifact(artifact)


def test_provenance_requires_fair_cache_and_batching_configuration() -> None:
    value = _provenance(
        tp=4,
        arm="kairyu",
        generation="generation",
        started_ns=1,
    )
    assert a6._provenance_valid(value, tp=4, arm="kairyu")
    for key in (
        "max_num_seqs",
        "cache_block_size",
        "cache_tokens",
        "kv_cache_dtype",
        "allocated_kv_cache_bytes_per_rank",
        "reserved_kv_cache_bytes_per_rank",
        "usable_kv_cache_bytes_per_rank",
        "chunked_prefill_enabled",
        "scheduler_policy",
        "max_model_len",
    ):
        changed = copy.deepcopy(value["value"])
        changed["configuration"][key] = None
        assert not a6._provenance_valid(
            a6.hashed_descriptor(changed),
            tp=4,
            arm="kairyu",
        )

    dirty = copy.deepcopy(value["value"])
    dirty["runtime"]["source_clean"] = False
    assert not a6._provenance_valid(
        a6.hashed_descriptor(dirty),
        tp=4,
        arm="kairyu",
    )

    malformed_commit = copy.deepcopy(value["value"])
    malformed_commit["runtime"]["build_commit"] = "not-a-full-commit"
    assert not a6._provenance_valid(
        a6.hashed_descriptor(malformed_commit),
        tp=4,
        arm="kairyu",
    )

    wrong_environment_hash = copy.deepcopy(value["value"])
    wrong_environment_hash["host"]["environment"]["artifact"]["sha256"] = "0" * 64
    assert not a6._provenance_valid(
        a6.hashed_descriptor(wrong_environment_hash),
        tp=4,
        arm="kairyu",
    )

    wrong_pci_assignment = copy.deepcopy(value["value"])
    wrong_pci_assignment["gpu"]["pci_bus_ids"][0] = "00000000:FF:00.0"
    assert not a6._provenance_valid(
        a6.hashed_descriptor(wrong_pci_assignment),
        tp=4,
        arm="kairyu",
    )

    wrong_container_image = copy.deepcopy(value["value"])
    launch = wrong_container_image["runtime"]["container_launch"]["value"]
    launch["image_id"] = f"sha256:{'9' * 64}"
    wrong_container_image["runtime"]["container_launch"] = (
        a6.hashed_descriptor(launch)
    )
    assert not a6._provenance_valid(
        a6.hashed_descriptor(wrong_container_image),
        tp=4,
        arm="kairyu",
    )

    missing_source_mount = copy.deepcopy(value["value"])
    launch = missing_source_mount["runtime"]["container_launch"]["value"]
    launch["mounts"] = [
        mount
        for mount in launch["mounts"]
        if mount["destination"] != "/app/kairyu"
    ]
    missing_source_mount["runtime"]["container_launch"] = (
        a6.hashed_descriptor(launch)
    )
    assert not a6._provenance_valid(
        a6.hashed_descriptor(missing_source_mount),
        tp=4,
        arm="kairyu",
    )

    different_yaml = copy.deepcopy(value["value"])
    source = different_yaml["configuration_source"]["value"]
    source["text"] = source["text"].replace(
        "pipeline_depth: 5",
        "pipeline_depth: 4",
    )
    source["text_sha256"] = a6.sha256_text(source["text"])
    source["parsed"]["engines"]["qwen3-32b"]["options"]["pipeline_depth"] = 4
    different_yaml["configuration_source"] = a6.hashed_descriptor(source)
    assert not a6._provenance_valid(
        a6.hashed_descriptor(different_yaml),
        tp=4,
        arm="kairyu",
    )

    stock = _provenance(
        tp=4,
        arm="vllm-stock",
        generation="stock-generation",
        started_ns=2,
    )
    wrong_capture_sizes = copy.deepcopy(stock["value"])
    wrong_capture_sizes["configuration"]["cuda_graph"]["capture_sizes"] = [
        1,
        2,
        4,
        8,
        16,
    ]
    assert not a6._provenance_valid(
        a6.hashed_descriptor(wrong_capture_sizes),
        tp=4,
        arm="vllm-stock",
    )

    unobserved_attention_backend = copy.deepcopy(stock["value"])
    startup = unobserved_attention_backend["runtime"]["startup_log"]["value"]
    startup["attention_backend_messages"] = ["Using FLASHINFER backend"]
    startup["attention_backend_line_sha256"] = a6.sha256_json(
        startup["attention_backend_messages"]
    )
    startup["resolved_markers"] = a6._vllm_startup_markers(
        startup["engine_config_message"],
        startup["attention_backend_messages"],
        startup["attention_version_messages"],
        startup["async_scheduling_messages"],
        tp=4,
    )
    unobserved_attention_backend["runtime"]["startup_log"] = (
        a6.hashed_descriptor(startup)
    )
    assert not a6._provenance_valid(
        a6.hashed_descriptor(unobserved_attention_backend),
        tp=4,
        arm="vllm-stock",
    )
