from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import ssl
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import dram_kv_tier_qwen as gate
from kairyu.engine.core.kv_tier_policy import (
    DramKVRuntimeIdentity,
    cpu_model_identity,
    engine_source_inventory,
    engine_source_sha256,
    load_crossover_policy,
)
from kairyu.engine.core.numa import GPU_LOCAL_FIRST_TOUCH_POLICY


def _digest(value: object) -> str:
    return gate.sha256_json(value)


def _source() -> dict[str, object]:
    inventory = engine_source_inventory()
    bound = {
        path: _digest(path)
        for path in sorted(gate.REQUIRED_SOURCE_PATHS)
        if not path.startswith("kairyu/")
    }
    bound.update(inventory)
    bound = dict(sorted(bound.items()))
    kairyu_paths = sorted(inventory)
    return {
        "commit": "a" * 40,
        "dirty": False,
        "source_kind": "git",
        "script_path": gate.SOURCE_PATH,
        "script_sha256": bound[gate.SOURCE_PATH],
        "bound_file_sha256": bound,
        "bound_files_sha256": gate.sha256_json(dict(sorted(bound.items()))),
        "kairyu_python_paths": kairyu_paths,
    }


def _checkpoint() -> dict[str, object]:
    weights = [
        {
            "file": f"model-{index:05d}-of-00017.safetensors",
            "bytes": index,
            "sha256": _digest(f"weight-{index}"),
        }
        for index in range(1, 18)
    ]
    config_sha = _digest("config")
    return {
        "model": gate.MODEL,
        "revision": gate.MODEL_REVISION,
        "architecture": {
            "model_type": "qwen3",
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_hidden_layers": gate.NUM_LAYERS,
            "num_attention_heads": 64,
            "num_key_value_heads": gate.NUM_KV_HEADS,
            "head_dim": gate.HEAD_DIM,
            "vocab_size": 151936,
        },
        "weights": weights,
        "weights_sha256": gate.sha256_json(weights),
        "pinned_weights_rollup": gate.EXPECTED_MODEL_WEIGHTS_ROLLUP,
        "metadata_sha256": {
            "config.json": config_sha,
            "tokenizer.json": _digest("tokenizer"),
        },
        "model_config_sha256": config_sha,
    }


def _attention_identity(
    resolved: str = "flashinfer",
    *,
    prefill_version: str | None = None,
    decode_version: str | None = None,
    aot_record_sha256: str | None = None,
) -> str:
    components = {
        "prefill": resolved,
        "decode": "flashinfer" if resolved != "torch" else "torch",
        "kv_mode": "paged-direct" if resolved != "torch" else "tensor-gather",
    }
    return json.dumps(
        {
            "requested": "auto",
            "resolved": resolved,
            "source": "hw_profile",
            "components": components,
            "component_versions": {
                "prefill": prefill_version
                or ("0.6.4" if resolved == "flashinfer" else "test-prefill-1"),
                "decode": decode_version
                or ("0.6.4" if resolved == "flashinfer" else "test-decode-1"),
            },
            "component_builds": {
                "flashinfer-jit-cache": {
                    "installed": True,
                    "version": "0.6.4",
                    "record_sha256": aot_record_sha256
                    or _digest("flashinfer-jit-cache-record"),
                }
            },
            "architecture": {
                "arch": "cuda",
                "device_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "sm": 120,
                "kernel_tier": "full",
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _runtime(tp_size: int) -> dict[str, object]:
    return {
        "tensor_parallel_size": tp_size,
        "engine_source_sha256": engine_source_sha256(),
        "page_size": gate.PAGE_SIZE,
        "kv_dtype": gate.DTYPE,
        "tier_backend": "cuda-pinned-dram",
        "recompute_path": "production-radix-model-prefill",
        "recompute_cache_disabled": True,
        "restore_path": gate.RESTORE_PATH,
        "control_backend": gate.CONTROL_BACKEND,
        "attention_backend": "flashinfer",
        "attention_backend_identity": _attention_identity(),
        "max_model_len": 8193,
        "max_num_batched_tokens": 2048,
        "host_memory_placement_policy": GPU_LOCAL_FIRST_TOUCH_POLICY,
        "os_page_size": os.sysconf("SC_PAGE_SIZE"),
        "pcie_max_link_speed": "32.0 GT/s PCIe",
        "pcie_max_link_width": 16,
        "pcie_current_link_width": 16,
        "local_numa_gpu_count": 2,
        "cpu_model": cpu_model_identity(),
        "python_version": platform.python_version(),
        "openssl_version": ssl.OPENSSL_VERSION,
        "chunked_prefill": True,
        "kv_layout_fingerprint": _digest(f"layout-tp{tp_size}"),
        "rank_pool_fingerprints": {
            str(rank): _digest(f"layout-tp{tp_size}-rank{rank}") for rank in range(tp_size)
        },
        "torch_version": "2.9.0",
        "torch_cuda_version": "13.0",
        "nccl_version": "2.29.7",
        "driver_version": "580.65.06",
    }


def _hardware(tp_size: int) -> dict[str, object]:
    slab_bytes = gate.expected_rank_bytes(tp_size, max(gate.PREFIX_LENGTHS))
    slab_pages = math.ceil(slab_bytes / 4096)
    return {
        "ranks": [
            {
                "rank": rank,
                "device": rank,
                "device_uuid": f"GPU-{rank:032d}",
                "pci_bus_id": f"0000:{16 + rank:02x}:00.0",
                "gpu_numa_node": rank // 2,
                "pcie_max_link_speed": "32.0 GT/s PCIe",
                "pcie_max_link_width": 16,
                "pcie_current_link_width": 16,
                "local_numa_gpu_count": 2,
                "host_numa_policy": GPU_LOCAL_FIRST_TOUCH_POLICY,
                "host_numa_node": rank // 2,
                "host_cpu_affinity": [rank * 2, rank * 2 + 1],
                "host_numa_page_counts": {str(rank // 2): slab_pages},
                "host_resident_pages": slab_pages,
                "host_slab_bytes": slab_bytes,
                "host_os_page_size": 4096,
                "host_full_slab_local_coverage": True,
                "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "gpu_sm": [12, 0],
            }
            for rank in range(tp_size)
        ],
        "nvidia_smi_topology": ["synthetic test topology"],
        "numa_topology": ["synthetic test NUMA map"],
    }


def _prompt_trace() -> dict[str, object]:
    return {
        "seed": 187,
        "max_prefix_tokens": max(gate.PREFIX_LENGTHS),
        "construction": "fixed-token-trace-v1",
        "full_token_ids_sha256": _digest("full-token-trace"),
        "prefix_token_ids_sha256": {
            str(prefix): _digest(["prefix", prefix]) for prefix in gate.PREFIX_LENGTHS
        },
    }


def _container(tp_size: int) -> dict[str, object]:
    return {
        "image_id": f"sha256:{_digest('image')}",
        "container_id": _digest(f"container-tp{tp_size}"),
        "command": ["python", "bench/dram_kv_tier_qwen.py", "measure"],
        "repo_digests": [f"kairyu@sha256:{_digest('image-repo')}"],
    }


def _cuda_span(elapsed_ns: int, stream_id: int) -> dict[str, object]:
    return {
        "start_recorded": True,
        "end_recorded": True,
        "complete": True,
        "stream_id": stream_id,
        "elapsed_ns": elapsed_ns,
    }


def _rank_events(
    *,
    tp_size: int,
    prefix: int,
    phase: str,
    transfer: bool,
    wall_ns: int | None,
) -> list[dict[str, object]]:
    expected_bytes = gate.expected_rank_bytes(tp_size, prefix)
    if phase == "offload":
        interval_names = ("offload_d2h",)
    elif phase == "restore":
        interval_names = ("restore_h2d", "next_token_model")
    else:
        interval_names = ("recompute_prefill", "next_token_model")
    ready_ns = min((wall_ns or 800_000) - 1, 700_000)
    result = []
    for rank in range(tp_size):
        numa_node = rank // 2
        staging = None
        if transfer:
            slab_bytes = gate.expected_rank_bytes(tp_size, max(gate.PREFIX_LENGTHS))
            resident_pages = math.ceil(slab_bytes / 4096)
            staging = {
                "pinned": True,
                "bytes": expected_bytes,
                "host_numa_policy": GPU_LOCAL_FIRST_TOUCH_POLICY,
                "host_numa_node": numa_node,
                "host_cpu_affinity": [rank * 2, rank * 2 + 1],
                "host_numa_page_counts": {str(numa_node): resident_pages},
                "host_resident_pages": resident_pages,
                "host_slab_bytes": slab_bytes,
                "host_os_page_size": 4096,
                "host_full_slab_local_coverage": True,
            }
        result.append(
            {
                "rank": rank,
                "device_uuid": f"GPU-{rank:032d}",
                "pci_bus_id": f"0000:{16 + rank:02x}:00.0",
                "gpu_numa_node": numa_node,
                "pool_fingerprint": _digest(f"layout-tp{tp_size}-rank{rank}"),
                "bytes": expected_bytes if transfer else 0,
                "kv_sha256": _digest([tp_size, prefix, rank, "kv"]),
                "cuda_ready_span": _cuda_span(ready_ns - rank, 100 + rank),
                "cuda_intervals": [
                    {
                        "name": name,
                        **_cuda_span(100_000 + index * 10_000 + rank, 200 + index),
                        **(
                            {
                                "scope": "pure-dma",
                                "telemetry": "tier-transfer.cuda_elapsed_ns",
                            }
                            if name in {"offload_d2h", "restore_h2d"}
                            else {}
                        ),
                    }
                    for index, name in enumerate(interval_names)
                ],
                "staging": staging,
            }
        )
    return result


def _ownership(prefix: int, arm: str, pair_order: list[str]) -> dict[str, object]:
    pages = prefix // gate.PAGE_SIZE
    host_before = (
        pages if arm == "restore" or (arm == "recompute" and pair_order[1] == "restore") else 0
    )
    host_after = pages if arm == "recompute" and pair_order[1] == "restore" else 0
    return {
        "before": {
            "destination_published": False,
            "host_resident_pages": host_before,
        },
        "pending": {"destination_published": False},
        "after": {
            "destination_published_pages": pages,
            "host_resident_pages": host_after,
            "live_transfer": False,
        },
    }


def _output(prefix: int) -> dict[str, object]:
    value = {
        "token_ids": [prefix % 151936],
        "text": f"token-{prefix}",
        "finish_reason": "length",
        "logprobs_sha256": _digest([prefix, "logprob"]),
    }
    return {"value": value, "sha256": gate.sha256_json(value)}


def _elapsed(
    *,
    prefix: int,
    repeat: int,
    arm: str,
    crossover: int,
    wins_at_and_above: int,
) -> int:
    recompute = 1_000_000 + prefix * 2_000
    if arm == "recompute":
        return recompute
    if prefix < crossover:
        return recompute * 6 // 5
    measured_repeat = max(repeat, 0)
    return recompute * 4 // 5 if measured_repeat < wins_at_and_above else recompute * 21 // 20


def _raw_rows(
    tp_size: int,
    *,
    crossover: int,
    wins_at_and_above: int = 8,
) -> list[dict[str, object]]:
    source = _source()
    checkpoint = _checkpoint()
    runtime = _runtime(tp_size)
    hardware = _hardware(tp_size)
    prompt_trace = _prompt_trace()
    run_id = f"test-tp{tp_size}"
    rows: list[dict[str, object]] = [
        {
            "row_type": "run",
            "schema_version": gate.SCHEMA_VERSION,
            "measurement_kind": gate.MEASUREMENT_KIND,
            "tp_size": tp_size,
            "run_id": run_id,
            "measurement_started_at": (
                "2026-07-31T00:00:00Z" if tp_size == 4 else "2026-07-31T02:00:00Z"
            ),
            "config": gate.formal_config(tp_size),
            "source": source,
            "checkpoint": checkpoint,
            "runtime": runtime,
            "hardware": hardware,
            "prompt_trace": prompt_trace,
            "container": _container(tp_size),
        }
    ]
    snapshot: dict[str, object] | None = None
    clock = 1_000_000_000
    for expected in gate.expected_schedule():
        prefix = int(expected["prefix_tokens"])
        repeat = int(expected["pair_index"])
        common = {
            "tp_size": tp_size,
            "prefix_tokens": prefix,
            "pair_index": repeat,
            "warmup": expected["warmup"],
            "pair_order": expected["pair_order"],
        }
        if expected["row_type"] == "snapshot":
            page_count = prefix // gate.PAGE_SIZE
            events = _rank_events(
                tp_size=tp_size,
                prefix=prefix,
                phase="offload",
                transfer=True,
                wall_ns=None,
            )
            digests = {str(event["rank"]): event["kv_sha256"] for event in events}
            snapshot = {
                "row_type": "snapshot",
                **common,
                "page_count": page_count,
                "token_ids_sha256": prompt_trace["prefix_token_ids_sha256"][str(prefix)],
                "source_page_ids": list(
                    range(gate.MAX_PREFIX_PAGES, gate.MAX_PREFIX_PAGES + page_count)
                ),
                "destination_page_ids": list(range(page_count)),
                "restore_path": gate.RESTORE_PATH,
                "control_backend": gate.CONTROL_BACKEND,
                "rank_events": events,
                "source_kv_sha256_by_rank": digests,
                "host_kv_sha256_by_rank": copy.deepcopy(digests),
                "source_invalidation_rank_receipts": [
                    {
                        "rank": rank,
                        "method": "cuda-zero-fill-plus-device-nonzero-check",
                        "fill_value": 0,
                        "page_count": page_count,
                        "bytes": gate.expected_rank_bytes(tp_size, prefix),
                        "page_ids_sha256": gate.sha256_json(
                            list(
                                range(
                                    gate.MAX_PREFIX_PAGES,
                                    gate.MAX_PREFIX_PAGES + page_count,
                                )
                            )
                        ),
                        "source_kv_sha256_before": digests[str(rank)],
                        "synchronized": True,
                        "all_zero": True,
                    }
                    for rank in range(tp_size)
                ],
                "async_submission": True,
                "ownership": {
                    "before": {
                        "source_device_owned_pages": page_count,
                        "tier_reserved_pages": 0,
                    },
                    "pending": {
                        "source_device_reusable": False,
                        "tier_published": False,
                    },
                    "after": {
                        "source_device_reusable": True,
                        "source_device_invalidated": True,
                        "tier_resident_pages": page_count,
                        "live_transfer": False,
                    },
                },
                "retry_count": 0,
                "error": None,
                "fallback_used": False,
            }
            rows.append(snapshot)
            continue
        assert snapshot is not None
        arm = str(expected["arm"])
        wall_ns = _elapsed(
            prefix=prefix,
            repeat=repeat,
            arm=arm,
            crossover=crossover,
            wins_at_and_above=wins_at_and_above,
        )
        events = _rank_events(
            tp_size=tp_size,
            prefix=prefix,
            phase=arm,
            transfer=arm == "restore",
            wall_ns=wall_ns,
        )
        digests = {str(event["rank"]): event["kv_sha256"] for event in events}
        rows.append(
            {
                "row_type": "sample",
                "arm": arm,
                **common,
                "controller_started_ns": clock,
                "controller_ended_ns": clock + wall_ns,
                "wall_elapsed_ns": wall_ns,
                "token_ids_sha256": snapshot["token_ids_sha256"],
                "restore_path": gate.RESTORE_PATH,
                "control_backend": gate.CONTROL_BACKEND,
                "rank_events": events,
                "source_kv_sha256_by_rank": copy.deepcopy(digests),
                "ready_kv_sha256_by_rank": copy.deepcopy(digests),
                "output": _output(prefix),
                "ownership": _ownership(prefix, arm, expected["pair_order"]),
                "retry_count": 0,
                "error": None,
                "fallback_used": False,
                "production_model_forward": True,
                "recompute_cache_hits": 0,
                "restored_from_dram": arm == "restore",
                "destination_page_ids_sha256": gate.sha256_json(snapshot["destination_page_ids"]),
            }
        )
        clock += wall_ns + 100_000
    rows.append(
        {
            "row_type": "run_end",
            "tp_size": tp_size,
            "run_id": run_id,
            "status": "complete",
            "errors": [],
            "measurement_ended_at": (
                "2026-07-31T01:00:00Z" if tp_size == 4 else "2026-07-31T03:00:00Z"
            ),
            "source": copy.deepcopy(source),
            "checkpoint": copy.deepcopy(checkpoint),
            "row_counts": {
                "snapshot_rows": 100,
                "sample_rows": 200,
                "warmup_sample_rows": 20,
                "measured_sample_rows": 180,
            },
        }
    )
    return [{**row, "row_index": index} for index, row in enumerate(rows)]


def _write_pair(tmp_path: Path, *, wins: int = 8) -> tuple[Path, Path]:
    paths = (tmp_path / "tp4.jsonl", tmp_path / "tp8.jsonl")
    gate.write_raw_shard(paths[0], _raw_rows(4, crossover=512, wins_at_and_above=wins))
    gate.write_raw_shard(paths[1], _raw_rows(8, crossover=1024, wins_at_and_above=wins))
    return paths


def test_fixed_schedule_bytes_and_statistical_rule() -> None:
    assert len(gate.expected_schedule()) + 2 == 302
    assert gate.expected_rank_bytes(4, 8192) == 512 * 1024**2
    assert gate.expected_rank_bytes(8, 8192) == 256 * 1024**2
    assert gate._sign_test_p_value(8) == pytest.approx(10 / 512)
    assert gate._sign_test_p_value(7) > 0.05
    measured = [row for row in gate.expected_schedule() if not row["warmup"]]
    assert measured[0]["pair_order"] == ["restore", "recompute"]
    repeat_one = next(row for row in measured if row["pair_index"] == 1)
    assert repeat_one["prefix_tokens"] == 8192
    assert repeat_one["pair_order"] == ["recompute", "restore"]


def test_crossover_uses_stable_suffix_without_interpolation() -> None:
    cells = [
        {"prefix_tokens": prefix, "passed": prefix in {64, 512, 1024, 2048, 4096, 8192}}
        for prefix in gate.PREFIX_LENGTHS
    ]
    crossover = gate.derive_crossover(cells)
    assert crossover == {
        "status": "bracketed",
        "min_restore_tokens": 512,
        "bracket": {
            "lower_exclusive_tokens": 256,
            "upper_inclusive_tokens": 512,
        },
        "interpolated": False,
    }


def test_complete_raw_replays_and_seals_runtime_profiles(tmp_path: Path) -> None:
    tp4, tp8 = _write_pair(tmp_path)
    manifest = gate.replay_raw(tp4, tp8, assert_gate=True)
    assert manifest["passed"] is True
    assert manifest["crossover"]["tp4"]["min_restore_tokens"] == 512
    assert manifest["crossover"]["tp8"]["min_restore_tokens"] == 1024
    assert manifest["cell_metrics"]["tp4"][5]["restore_wins"] == 8

    artifact = tmp_path / "artifact"
    gate.assemble_artifact(tp4, tp8, artifact, assert_gate=True)
    assert gate.verify_artifact(artifact, assert_gate=True) == manifest
    profile = json.loads((artifact / gate.PROFILE_NAMES[4]).read_text())
    assert set(profile) == {
        "schema_version",
        "runtime_identity",
        "runtime_fingerprint",
        "samples",
        "policy",
        "evidence_sha256",
        "passed",
    }
    assert profile["policy"] == {
        "rule": "median_lt_1_and_wins_at_least_8_of_9",
        "min_restore_tokens": 512,
    }
    assert len(profile["samples"]) == 90
    assert {row["prefix_tokens"] for row in profile["samples"]} == set(gate.PREFIX_LENGTHS)
    identity = DramKVRuntimeIdentity.from_json(profile["runtime_identity"])
    assert identity.attention_backend_identity == _attention_identity()
    assert identity.engine_source_sha256 == engine_source_sha256()
    assert identity.max_num_batched_tokens == 2048
    assert identity.host_memory_placement_policy == GPU_LOCAL_FIRST_TOUCH_POLICY
    assert identity.os_page_size == os.sysconf("SC_PAGE_SIZE")
    assert identity.pcie_max_link_speed == "32.0 GT/s PCIe"
    assert identity.pcie_max_link_width == 16
    assert identity.pcie_current_link_width == 16
    assert identity.local_numa_gpu_count == 2
    assert identity.cpu_model == cpu_model_identity()
    assert identity.python_version == platform.python_version()
    assert identity.openssl_version == ssl.OPENSSL_VERSION
    loaded = load_crossover_policy(
        artifact / gate.PROFILE_NAMES[4],
        expected_identity=identity,
    )
    assert loaded.min_restore_tokens == 512


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "attention_backend_identity",
            _attention_identity("flashattention3"),
            "TP runtime differs at attention_backend_identity",
        ),
        (
            "max_num_batched_tokens",
            1024,
            "TP runtime differs at max_num_batched_tokens",
        ),
        (
            "attention_backend_identity",
            _attention_identity(
                prefill_version="0.7.0",
                decode_version="0.7.0",
            ),
            "TP runtime differs at attention_backend_identity",
        ),
        (
            "attention_backend_identity",
            _attention_identity(aot_record_sha256=_digest("different-aot-wheel")),
            "TP runtime differs at attention_backend_identity",
        ),
        (
            "pcie_current_link_width",
            8,
            "TP runtime differs at pcie_current_link_width",
        ),
        (
            "cpu_model",
            "different-cpu-model",
            "TP runtime differs at cpu_model",
        ),
    ],
    ids=(
        "attention-backend",
        "chunk-budget",
        "backend-package-version",
        "backend-aot-build",
        "pcie-width",
        "cpu-model",
    ),
)
def test_tp_profiles_cannot_share_a_crossover_across_execution_policy_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    tp4, tp8 = _write_pair(tmp_path)
    rows = gate.read_raw_shard(tp8)
    rows[0]["runtime"][field] = value
    gate.write_raw_shard(tp8, rows)
    with pytest.raises(gate.EvidenceError, match=message):
        gate.replay_raw(tp4, tp8)


def test_runtime_rejects_a_noncanonical_attention_backend_identity(
    tmp_path: Path,
) -> None:
    tp4, tp8 = _write_pair(tmp_path)
    rows = gate.read_raw_shard(tp4)
    parsed = json.loads(rows[0]["runtime"]["attention_backend_identity"])
    noncanonical = json.dumps(parsed, sort_keys=False, indent=2)
    rows[0]["runtime"]["attention_backend_identity"] = noncanonical
    gate.write_raw_shard(tp4, rows)
    with pytest.raises(gate.EvidenceError, match="not canonical"):
        gate.replay_raw(tp4, tp8)


def test_seven_of_nine_does_not_publish_a_crossover(tmp_path: Path) -> None:
    tp4, tp8 = _write_pair(tmp_path, wins=7)
    manifest = gate.replay_raw(tp4, tp8)
    assert manifest["passed"] is False
    assert manifest["crossover"]["tp4"]["status"] == "none_in_measured_range"
    with pytest.raises(gate.EvidenceError, match="stable_measured_crossover"):
        gate.replay_raw(tp4, tp8, assert_gate=True)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda rows: rows[2]["rank_events"][0]["staging"].__setitem__(
                "host_full_slab_local_coverage", False
            ),
            "complete pinned slab",
        ),
        (
            lambda rows: rows[2]["rank_events"][0].__setitem__("bytes", 1),
            "byte count differs",
        ),
        (
            lambda rows: rows[3].__setitem__("production_model_forward", False),
            "production model forward",
        ),
        (
            lambda rows: rows[1].__setitem__("destination_page_ids", rows[1]["source_page_ids"]),
            "source-high/destination-zero remap",
        ),
        (
            lambda rows: rows[1]["source_invalidation_rank_receipts"][0].__setitem__(
                "all_zero", False
            ),
            "source HBM invalidation receipt",
        ),
        (
            lambda rows: rows[2].__setitem__("restore_path", "direct-tier-restore"),
            "production Radix/DistTP Gloo",
        ),
        (
            lambda rows: rows[2]["rank_events"][0]["cuda_intervals"][0].pop(
                "scope"
            ),
            "exact tier DMA telemetry",
        ),
    ],
)
def test_binding_numa_bytes_model_and_remap_checks(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    tp4, tp8 = _write_pair(tmp_path)
    rows = gate.read_raw_shard(tp4)
    mutate(rows)
    gate.write_raw_shard(tp4, rows)
    with pytest.raises(gate.EvidenceError, match=message):
        gate.replay_raw(tp4, tp8)


def test_output_or_kv_mismatch_is_not_accepted(tmp_path: Path) -> None:
    tp4, tp8 = _write_pair(tmp_path)
    rows = gate.read_raw_shard(tp4)
    restore = next(
        row
        for row in rows
        if row.get("row_type") == "sample"
        and row.get("arm") == "restore"
        and row.get("pair_index") == 0
    )
    output = copy.deepcopy(restore["output"]["value"])
    output["text"] = "different"
    restore["output"] = {"value": output, "sha256": gate.sha256_json(output)}
    gate.write_raw_shard(tp4, rows)
    with pytest.raises(gate.EvidenceError, match="output parity"):
        gate.replay_raw(tp4, tp8)


def test_only_rank0_ready_span_must_fit_the_rank0_controller_wall(
    tmp_path: Path,
) -> None:
    tp4, tp8 = _write_pair(tmp_path)
    rows = gate.read_raw_shard(tp4)
    sample = next(row for row in rows if row.get("row_type") == "sample")
    wall_ns = sample["wall_elapsed_ns"]

    # A follower can start its event before rank 0 returns from the leading
    # barrier.  Its local CUDA interval is still valid even when that interval
    # is wider than rank 0's independently sampled controller wall.
    sample["rank_events"][1]["cuda_ready_span"]["elapsed_ns"] = wall_ns + 1
    gate.write_raw_shard(tp4, rows)
    assert gate.replay_raw(tp4, tp8)["passed"] is True

    # Rank 0 records its event after its own CPU start and synchronizes it before
    # its own CPU end, so the same mismatch on rank 0 is a real contradiction.
    sample["rank_events"][0]["cuda_ready_span"]["elapsed_ns"] = wall_ns + 1
    gate.write_raw_shard(tp4, rows)
    with pytest.raises(gate.EvidenceError, match="rank-0 CUDA ready span"):
        gate.replay_raw(tp4, tp8)


def test_source_must_remain_stable_across_measurement(tmp_path: Path) -> None:
    tp4, tp8 = _write_pair(tmp_path)
    rows = gate.read_raw_shard(tp4)
    rows[-1]["source"]["commit"] = "b" * 40
    gate.write_raw_shard(tp4, rows)
    with pytest.raises(gate.EvidenceError, match="source changed"):
        gate.replay_raw(tp4, tp8)


def test_runtime_source_rollup_must_match_the_full_provenance_inventory(
    tmp_path: Path,
) -> None:
    tp4, tp8 = _write_pair(tmp_path)
    rows = gate.read_raw_shard(tp4)
    source = rows[0]["source"]
    model_path = next(
        path for path in source["kairyu_python_paths"] if path.startswith("kairyu/models/")
    )
    source["bound_file_sha256"][model_path] = _digest("changed-unlisted-model")
    source["bound_files_sha256"] = gate.sha256_json(
        dict(sorted(source["bound_file_sha256"].items()))
    )
    gate.write_raw_shard(tp4, rows)

    with pytest.raises(gate.EvidenceError, match="installed Kairyu source differs"):
        gate.replay_raw(tp4, tp8)


def test_verify_rejects_manifest_and_profile_tampering(tmp_path: Path) -> None:
    tp4, tp8 = _write_pair(tmp_path)
    artifact = tmp_path / "artifact"
    gate.assemble_artifact(tp4, tp8, artifact)

    manifest_path = artifact / gate.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["passed"] = False
    manifest_path.write_text(f"{gate.canonical_json(manifest)}\n")
    with pytest.raises(gate.EvidenceError, match="manifest differs"):
        gate.verify_artifact(artifact)

    gate.assemble_artifact(tp4, tp8, artifact)
    profile_path = artifact / gate.PROFILE_NAMES[8]
    profile = json.loads(profile_path.read_text())
    profile["policy"]["min_restore_tokens"] = 16
    profile_path.write_text(f"{gate.canonical_json(profile)}\n")
    with pytest.raises(gate.EvidenceError, match="runtime profile differs"):
        gate.verify_artifact(artifact)


def test_jsonl_requires_complete_framing(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"row_index":0}')
    with pytest.raises(gate.EvidenceError, match="framing"):
        gate.read_raw_shard(path)


def test_live_container_identity_is_full_exact_and_file_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "a" * 64
    identity_file = tmp_path / "container-id"
    identity_file.write_text(f"{container_id}\n")

    assert (
        gate._resolve_container_id(
            container_id=None,
            container_id_file=identity_file,
        )
        == container_id
    )
    with pytest.raises(gate.EvidenceError, match="argument and file differ"):
        gate._resolve_container_id(
            container_id="b" * 64,
            container_id_file=identity_file,
        )
    with pytest.raises(gate.EvidenceError, match="full 64-hex"):
        gate._resolve_container_id(container_id="a" * 12, container_id_file=None)

    monkeypatch.setattr(gate.socket, "gethostname", lambda: container_id[:12])
    container = gate._container_provenance_live(
        container_id=container_id,
        image_id=f"sha256:{'c' * 64}",
        repo_digests=(),
        command=("python", gate.SOURCE_PATH, "run"),
    )
    assert container["container_id"] == container_id
    assert container["repo_digests"] == []
    assert container["hostname"] == container_id[:12]


def test_live_cpu_and_checksum_contract_helpers_fail_closed() -> None:
    checksums = (_digest("page-0"), _digest("page-1"))
    assert gate._aggregate_checksums(checksums) == gate.sha256_json(list(checksums))
    assert gate._aggregate_checksums(checksums) != gate._aggregate_checksums(
        tuple(reversed(checksums))
    )
    with pytest.raises(gate.EvidenceError, match="invalid KV checksums"):
        gate._aggregate_checksums(("not-a-digest",))

    from kairyu.engine.core.radix_kv import _extend_prefix_hash_chain

    tokens = tuple(range(64))
    _, _, _, expected = _extend_prefix_hash_chain(
        hashlib.sha256(b"("),
        0,
        tokens,
        gate.PAGE_SIZE,
    )
    assert gate._radix_tier_keys(tokens, prefix_tokens=64) == expected

    metrics = SimpleNamespace(
        direction="restore",
        page_count=4,
        bytes_transferred=4096,
        stream_id=77,
        cuda_elapsed_ns=1234,
    )
    assert gate._dma_span_from_transfer(
        metrics,
        direction="restore",
        page_count=4,
        bytes_transferred=4096,
    ) == {
        "start_recorded": True,
        "end_recorded": True,
        "complete": True,
        "stream_id": 77,
        "elapsed_ns": 1234,
        "scope": "pure-dma",
        "telemetry": "tier-transfer.cuda_elapsed_ns",
    }
    with pytest.raises(gate.EvidenceError, match="direction differs"):
        gate._dma_span_from_transfer(
            metrics,
            direction="offload",
            page_count=4,
            bytes_transferred=4096,
        )


def test_live_receipts_preserve_rank_order_and_ba_host_ownership() -> None:
    prefix_tokens = 64
    page_count = prefix_tokens // gate.PAGE_SIZE
    pair_order = ("recompute", "restore")
    source_digests = {"0": _digest("rank-0"), "1": _digest("rank-1")}
    receipts = [
        {
            "rank": rank,
            "rank_event": {"rank": rank},
            "kv_sha256": source_digests[str(rank)],
            "output_token_id": 42,
            "host_resident_pages_before": page_count,
            "host_resident_pages_after": page_count,
            "destination_published_pages": page_count,
            "recompute_cache_hits": 0,
            "restored_from_dram": False,
            "control_operations": [],
        }
        for rank in range(2)
    ]

    class _Tokenizer:
        @staticmethod
        def decode(token_ids: tuple[int, ...]) -> str:
            assert token_ids == (42,)
            return "answer"

    row = gate._sample_row_from_receipts(
        tp_size=2,
        prefix_tokens=prefix_tokens,
        pair_index=1,
        warmup=False,
        pair_order=pair_order,
        arm="recompute",
        destination_pages=tuple(range(page_count)),
        token_ids_sha256=_digest("prefix"),
        source_digests=source_digests,
        controller_started_ns=100,
        controller_ended_ns=250,
        receipts=receipts,
        tokenizer=_Tokenizer(),
    )
    assert row["wall_elapsed_ns"] == 150
    assert row["ownership"]["before"]["host_resident_pages"] == page_count
    assert row["ownership"]["after"]["host_resident_pages"] == page_count
    assert row["output"]["value"]["token_ids"] == [42]

    receipts[1]["output_token_id"] = 43
    with pytest.raises(gate.EvidenceError, match="one output token"):
        gate._sample_row_from_receipts(
            tp_size=2,
            prefix_tokens=prefix_tokens,
            pair_index=1,
            warmup=False,
            pair_order=pair_order,
            arm="recompute",
            destination_pages=tuple(range(page_count)),
            token_ids_sha256=_digest("prefix"),
            source_digests=source_digests,
            controller_started_ns=100,
            controller_ended_ns=250,
            receipts=receipts,
            tokenizer=_Tokenizer(),
        )


def test_run_cli_requires_explicit_inputs_and_routes_one_tp_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    container_id = "d" * 64
    identity_file = tmp_path / "container-id"
    identity_file.write_text(container_id)
    model_path = tmp_path / "model"
    output = tmp_path / "tp4.jsonl"
    captured: dict[str, object] = {}

    monkeypatch.setattr(gate.socket, "gethostname", lambda: container_id[:12])

    def _fake_run_live_shard(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"passed_contract_validation": True}

    monkeypatch.setattr(gate, "_run_live_shard", _fake_run_live_shard)
    assert (
        gate.main(
            (
                "run",
                "--tp",
                "4",
                "--model-path",
                str(model_path),
                "--output",
                str(output),
                "--image-id",
                f"sha256:{'e' * 64}",
                "--container-id-file",
                str(identity_file),
            )
        )
        == 0
    )
    assert captured["tp_size"] == 4
    assert captured["model_path"] == model_path
    assert captured["output"] == output
    assert captured["container"]["container_id"] == container_id
    assert json.loads(capsys.readouterr().out) == {"passed_contract_validation": True}
