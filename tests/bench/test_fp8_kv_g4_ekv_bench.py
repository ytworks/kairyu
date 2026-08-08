"""Offline contracts for the replayable G4 E-KV FP8 correctness bake."""

from __future__ import annotations

import base64
import copy
import subprocess
from collections import Counter
from pathlib import Path

import pytest
import torch

from bench import fp8_kv_g4_ekv_bench as gate
from kairyu.engine.core.kv_pool import PagedKVPool


def _source() -> dict[str, object]:
    return {
        "commit": "a" * 40,
        "tracked_dirty": False,
        "files": {path: "b" * 64 for path in gate.BOUND_SOURCE_FILES},
    }


def _compiler() -> dict[str, object]:
    shared_objects = [
        {
            "path": (
                "0.6.14/120f/cached_ops/"
                "batch_prefill_dtype_kv_bf16_head_dim_qk_128/kernel.so"
            ),
            "size": 1,
            "sha256": "c" * 64,
        },
        {
            "path": (
                "0.6.14/120f/cached_ops/"
                "batch_prefill_dtype_kv_e4m3_head_dim_qk_128/kernel.so"
            ),
            "size": 1,
            "sha256": "d" * 64,
        },
    ]
    return {
        "mode": "container-with-mounted-source-jit-runtime",
        "python_executable": "/runtime/.venv/bin/python",
        "nvcc_path": "/usr/local/cuda/bin/nvcc",
        "nvcc_sha256": "e" * 64,
        "nvcc_version": "Cuda compilation tools, release 13.3",
        "host_compiler": {
            "gcc": {
                "path": "/usr/bin/gcc",
                "sha256": "3" * 64,
                "version": "gcc 13.3.0",
            },
            "g++": {
                "path": "/usr/bin/g++",
                "sha256": "4" * 64,
                "version": "g++ 13.3.0",
            },
            "cc1plus": {
                "path": "/usr/libexec/gcc/13/cc1plus",
                "sha256": "5" * 64,
                "version": "GNU C++ 13.3.0",
            },
        },
        "HOME": "/runtime-home",
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_EXTENSIONS_DIR": "/jit/torch",
        "FLASHINFER_WORKSPACE_BASE": "/flashinfer-workspace",
        "FLASHINFER_WORKSPACE_FOLDER": None,
        "resolved_flashinfer_cache_root": (
            "/flashinfer-workspace/.cache/flashinfer"
        ),
        "flashinfer_shared_objects": shared_objects,
        "flashinfer_shared_objects_rollup_sha256": gate.sha256_json(
            shared_objects
        ),
    }


def _environment() -> dict[str, object]:
    return {
        "packages": {
            "python": "3.12.3",
            "torch": "2.12.1+cu130",
            "flashinfer": "0.6.14",
            "triton": "3.5.0",
        },
        "cuda": {
            "torch_cuda_version": "13.0",
            "cudnn_version": 91002,
            "visible_device_count": 1,
        },
        "gpu": {
            "torch_name": gate.EXPECTED_GPU_NAME,
            "compute_capability": [12, 0],
            "sm": 120,
            "total_memory_bytes": 96 * 1024**3,
            "nvidia_smi": [
                {
                    "index": 7,
                    "uuid": "GPU-" + "f" * 36,
                    "pci_bus_id": "00000000:D8:00.0",
                    "name": gate.EXPECTED_GPU_NAME,
                    "driver_version": "590.44",
                    "memory_total_mib": 97_887,
                }
            ],
        },
        "visibility": {
            "CUDA_VISIBLE_DEVICES": "0",
            "NVIDIA_VISIBLE_DEVICES": "GPU-" + "f" * 36,
        },
        "compiler": _compiler(),
        "attention_backend": {
            "requested": "flashinfer",
            "resolved": "flashinfer",
            "source": "env",
            "components": {
                "prefill": "flashinfer",
                "decode": "flashinfer",
                "kv_mode": "paged-direct",
            },
            "architecture": {"sm": 120},
        },
    }


def _checkpoint() -> dict[str, object]:
    weights = [dict(row) for row in gate.EXPECTED_WEIGHT_ROWS]
    return {
        "model": gate.MODEL,
        "revision": gate.MODEL_REVISION,
        "config_sha256": gate.EXPECTED_CONFIG_SHA256,
        "index_sha256": gate.EXPECTED_INDEX_SHA256,
        "tokenizer_sha256": gate.EXPECTED_TOKENIZER_SHA256,
        "weights": weights,
        "weights_rollup_sha256": gate.sha256_json(weights),
    }


def _update_end_counts(rows: list[dict[str, object]]) -> None:
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
    environment = _environment()
    calibration = gate._load_calibration()
    rows: list[dict[str, object]] = [
        {
            "schema_version": gate.SCHEMA_VERSION,
            "type": "run",
            "created_at": "2026-07-31T00:00:00+00:00",
            "config": gate.formal_config(),
            "image_id": "sha256:" + "1" * 64,
            "source_start": source,
            "checkpoint": _checkpoint(),
            "environment_start": environment,
        }
    ]
    for index, length in enumerate(gate.PROMPT_LENGTHS):
        token_id = 100 + index
        piece = f" p{index}"
        text = piece * length
        token_ids = [token_id] * length
        rows.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "type": "prompt",
                "prompt_id": f"long-{length}",
                "length": length,
                "piece_token_id": token_id,
                "piece_text": piece,
                "text": text,
                "text_sha256": gate.sha256_text(text),
                "token_ids": token_ids,
                "token_ids_sha256": gate.sha256_json(token_ids),
            }
        )
    for length in gate.PROMPT_LENGTHS:
        tokens = list(range(gate.OUTPUT_TOKENS))
        text = "fixture output"
        for arm in gate.ARMS:
            rows.append(
                {
                    "schema_version": gate.SCHEMA_VERSION,
                    "type": "output",
                    "prompt_id": f"long-{length}",
                    "prompt_length": length,
                    "arm": arm,
                    "kv_dtype": (
                        "bfloat16"
                        if arm == "bf16"
                        else "float8_e4m3fn"
                    ),
                    "k_scale": None if arm == "bf16" else "per-layer",
                    "v_scale": None if arm == "bf16" else "per-layer",
                    "calibration_sha256": (
                        None if arm == "bf16" else calibration.sha256
                    ),
                    "output_token_ids": tokens,
                    "output_token_ids_sha256": gate.sha256_json(tokens),
                    "selected_logprobs": [-1.0] * gate.OUTPUT_TOKENS,
                    "finish_reason": "length",
                    "output_text": text,
                    "output_text_sha256": gate.sha256_text(text),
                }
            )
    for (phase, kind), expected in sorted(
        gate._expected_audit_counts().items()
    ):
        rows.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "type": "write_audit",
                "phase": phase,
                "kind": kind,
                "input_dtype": "bfloat16",
                "storage_dtype": "float8_e4m3fn",
                "scale_mode": "per-layer-kv",
                "calibration_sha256": calibration.sha256,
                "calls": expected["calls"],
                "values": expected["values"],
                "input_nonfinite": 0,
                "input_overrange": 0,
                "input_dtype_mismatch": 0,
                "stored_nonfinite": 0,
                "stored_byte_mismatch": 0,
                "dequant_error_violation": 0,
                "max_abs_input": 1.0,
                "max_abs_error": 0.0,
                "max_bound_excess": 0.0,
            }
        )
    bf16_raw = torch.zeros(
        (gate.NUM_KV_HEADS, gate.HEAD_DIM), dtype=torch.bfloat16
    ).view(torch.uint8).numpy().tobytes()
    fp8_raw = torch.zeros(
        (gate.NUM_KV_HEADS, gate.HEAD_DIM), dtype=torch.float8_e4m3fn
    ).view(torch.uint8).numpy().tobytes()
    for length in gate.PROMPT_LENGTHS:
        for layer in gate.SAMPLE_LAYERS:
            for position in gate.sample_positions(length):
                for kind in ("k", "v"):
                    rows.append(
                        {
                            "schema_version": gate.SCHEMA_VERSION,
                            "type": "cache_sample",
                            "prompt_id": f"long-{length}",
                            "prompt_length": length,
                            "layer": layer,
                            "position": position,
                            "kind": kind,
                            "shape": [gate.NUM_KV_HEADS, gate.HEAD_DIM],
                            "bf16_dtype": "bfloat16",
                            "bf16_base64": base64.b64encode(
                                bf16_raw
                            ).decode(),
                            "bf16_sha256": gate.sha256_bytes(bf16_raw),
                            "fp8_dtype": "float8_e4m3fn",
                            "fp8_scale": (
                                calibration.k_scales[layer]
                                if kind == "k"
                                else calibration.v_scales[layer]
                            ),
                            "fp8_base64": base64.b64encode(fp8_raw).decode(),
                            "fp8_sha256": gate.sha256_bytes(fp8_raw),
                        }
                    )
    rows.append(
        {
            "schema_version": gate.SCHEMA_VERSION,
            "type": "run_end",
            "source_end": source,
            "environment_end": environment,
            "row_counts_before_end": {},
        }
    )
    _update_end_counts(rows)
    return rows


def test_formal_config_has_sixteen_unique_in_range_samples() -> None:
    config = gate.formal_config()
    for length in gate.PROMPT_LENGTHS:
        positions = gate.sample_positions(length)
        assert len(positions) == gate.SAMPLE_POSITIONS_PER_PROMPT
        assert len(set(positions)) == gate.SAMPLE_POSITIONS_PER_PROMPT
        assert all(0 <= position < length for position in positions)
        assert config["sample_positions"][str(length)] == list(positions)


def test_compiler_snapshot_uses_nonwriting_cc1plus_version_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tools = {
        name: tmp_path / name
        for name in ("nvcc", "gcc", "g++", "cc1plus")
    }
    for path in tools.values():
        path.write_text(path.name, encoding="utf-8")
    cache_root = tmp_path / "workspace" / ".cache" / "flashinfer"
    cache_root.mkdir(parents=True)
    (cache_root / "kernel.so").write_bytes(b"kernel")
    calls: list[tuple[str, ...]] = []

    def fake_run(
        command: tuple[str, ...],
        **_: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[-1] == "-print-prog-name=cc1plus":
            stdout = str(tools["cc1plus"])
        elif Path(command[0]).name == "cc1plus":
            stdout = (
                "GNU C++17 13.3.0\n\n"
                "Time variable usr sys wall GGC\n"
                " TOTAL 0.00 0.00 0.01 2469k"
            )
        elif Path(command[0]).name == "nvcc":
            stdout = "Cuda compilation tools, release 13.3"
        else:
            stdout = f"{Path(command[0]).name} 13.3.0"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(
        gate.shutil,
        "which",
        lambda name: str(tools[name]),
    )
    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(
        "FLASHINFER_WORKSPACE_BASE",
        str(tmp_path / "workspace"),
    )
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "torch"))

    snapshot = gate._compiler_snapshot()

    cc1plus = str(tools["cc1plus"].resolve())
    assert (
        cc1plus,
        "-version",
        "-fsyntax-only",
        "/dev/null",
    ) in calls
    assert (cc1plus, "--version") not in calls
    assert snapshot["host_compiler"]["cc1plus"]["version"] == (
        "GNU C++17 13.3.0"
    )


def test_valid_raw_passes_write_verify_and_replay(
    tmp_path: Path,
    valid_rows: list[dict[str, object]],
) -> None:
    manifest = gate.write_artifact(tmp_path, valid_rows)
    assert manifest["verdict"] == "PASS"
    assert gate.verify_artifact(tmp_path) == manifest
    assert gate.replay_artifact(tmp_path) == manifest


def test_environment_stability_ignores_only_cc1plus_timing(
    valid_rows: list[dict[str, object]],
) -> None:
    rows = copy.deepcopy(valid_rows)
    rows[-1]["environment_end"] = copy.deepcopy(
        rows[-1]["environment_end"]
    )
    start_version = rows[0]["environment_start"]["compiler"][
        "host_compiler"
    ]["cc1plus"]["version"]
    rows[0]["environment_start"]["compiler"]["host_compiler"]["cc1plus"][
        "version"
    ] = f"{start_version}\nTime variable wall\n TOTAL 0.01"
    rows[-1]["environment_end"]["compiler"]["host_compiler"]["cc1plus"][
        "version"
    ] = f"{start_version}\nTime variable wall\n TOTAL 0.00"

    manifest = gate.recompute_manifest(rows, raw_sha256="2" * 64)

    assert manifest["checks"]["sm120_flashinfer_environment_stable"] is True


def test_logprob_gate_compares_only_the_common_output_prefix(
    valid_rows: list[dict[str, object]],
) -> None:
    rows = copy.deepcopy(valid_rows)
    row = next(
        row
        for row in rows
        if row.get("type") == "output"
        and row.get("prompt_length") == gate.PROMPT_LENGTHS[0]
        and row.get("arm") == "fp8_e4m3"
    )
    row["output_token_ids"] = list(row["output_token_ids"])
    row["output_token_ids"][2] = 999
    row["output_token_ids_sha256"] = gate.sha256_json(
        row["output_token_ids"]
    )
    row["selected_logprobs"][2:] = [-100.0] * (gate.OUTPUT_TOKENS - 2)

    manifest = gate.recompute_manifest(rows, raw_sha256="2" * 64)
    metric = next(
        item
        for item in manifest["metrics"]["outputs"]
        if item["prompt_length"] == gate.PROMPT_LENGTHS[0]
    )

    assert manifest["checks"]["exact_output_tokens_and_stopping"] is False
    assert manifest["checks"]["selected_logprob_delta_at_most_0_25"] is True
    assert metric["common_prefix_tokens"] == 2
    assert metric["compared_logprob_positions"] == 2
    assert metric["max_selected_logprob_abs_delta"] == 0.0


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("output-token", "exact_output_tokens_and_stopping"),
        ("logprob", "selected_logprob_delta_at_most_0_25"),
        ("overrange", "all_fp8_writes_audited"),
        ("cache", "cache_sample_nrmse_at_most_0_05"),
        ("dirty-source", "clean_source_stable"),
        ("failure", "no_runtime_failure"),
    ],
)
def test_binding_tamper_is_fail_closed(
    valid_rows: list[dict[str, object]],
    mutation: str,
    failed_check: str,
) -> None:
    rows = copy.deepcopy(valid_rows)
    if mutation == "output-token":
        row = next(
            row
            for row in rows
            if row.get("type") == "output"
            and row.get("arm") == "fp8_e4m3"
        )
        row["output_token_ids"][0] = gate.OUTPUT_TOKENS + 1
        row["output_token_ids_sha256"] = gate.sha256_json(
            row["output_token_ids"]
        )
    elif mutation == "logprob":
        row = next(
            row
            for row in rows
            if row.get("type") == "output"
            and row.get("arm") == "fp8_e4m3"
        )
        row["selected_logprobs"][0] = -0.7
    elif mutation == "overrange":
        row = next(row for row in rows if row.get("type") == "write_audit")
        row["input_overrange"] = 1
    elif mutation == "cache":
        row = next(row for row in rows if row.get("type") == "cache_sample")
        raw = bytearray(base64.b64decode(row["fp8_base64"]))
        raw[0] = 0x38
        row["fp8_base64"] = base64.b64encode(raw).decode()
        row["fp8_sha256"] = gate.sha256_bytes(raw)
    elif mutation == "dirty-source":
        rows[0]["source_start"]["tracked_dirty"] = True
    else:
        rows.insert(
            -1,
            {
                "schema_version": gate.SCHEMA_VERSION,
                "type": "failure",
                "error_type": "RuntimeError",
                "message": "fixture failure",
            },
        )
        _update_end_counts(rows)
    manifest = gate.recompute_manifest(rows, raw_sha256="2" * 64)
    assert manifest["verdict"] == "FAIL"
    assert manifest["checks"][failed_check] is False


def test_failure_artifact_is_retained(
    tmp_path: Path,
    valid_rows: list[dict[str, object]],
) -> None:
    rows = copy.deepcopy(valid_rows)
    rows.insert(
        -1,
        {
            "schema_version": gate.SCHEMA_VERSION,
            "type": "failure",
            "error_type": "RuntimeError",
            "message": "retained",
        },
    )
    _update_end_counts(rows)
    assert gate.write_artifact(tmp_path, rows)["verdict"] == "FAIL"
    assert (tmp_path / gate.RAW_NAME).is_file()
    assert (tmp_path / gate.MANIFEST_NAME).is_file()


def test_satfinite_oracle_does_not_conflate_overrange_with_bytes() -> None:
    pool = PagedKVPool(
        num_layers=1,
        num_pages=1,
        page_size=gate.PAGE_SIZE,
        num_kv_heads=gate.NUM_KV_HEADS,
        head_dim=gate.HEAD_DIM,
        dtype=torch.float8_e4m3fn,
    )
    auditor = gate._FP8WriteAuditor(pool, gate._load_calibration().sha256)
    auditor.install()
    source = torch.full(
        (1, gate.NUM_KV_HEADS, gate.HEAD_DIM),
        500.0,
        dtype=torch.bfloat16,
    )
    try:
        pool.write_ragged(
            0,
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([0]),
            torch.tensor([0]),
            source,
            source,
            torch.tensor([0]),
        )
    finally:
        auditor.restore()
    row = next(
        row
        for row in auditor.rows()
        if row["phase"] == "ragged_prefill" and row["kind"] == "k"
    )
    assert row["input_overrange"] == gate.NUM_KV_HEADS * gate.HEAD_DIM
    assert row["stored_nonfinite"] == 0
    assert row["stored_byte_mismatch"] == 0
    assert torch.isfinite(pool.k.to(torch.float32)).all()


def test_dequant_audit_absorbs_only_fp64_boundary_noise() -> None:
    scale = 0.0035156250000000005
    pool = PagedKVPool(
        num_layers=1,
        num_pages=1,
        page_size=1,
        num_kv_heads=1,
        head_dim=1,
        dtype=torch.float8_e4m3fn,
        k_scales=[scale],
        v_scales=[scale],
    )
    auditor = gate._FP8WriteAuditor(pool, gate._load_calibration().sha256)
    source = torch.tensor([5.14984130859375e-05], dtype=torch.bfloat16)
    actual = (source.float() / scale).to(torch.float8_e4m3fn)

    auditor._audit("ragged_prefill", 0, "k", source, actual)
    row = next(
        row
        for row in auditor.rows()
        if row["phase"] == "ragged_prefill" and row["kind"] == "k"
    )
    assert row["stored_byte_mismatch"] == 0
    assert row["dequant_error_violation"] == 0
    assert row["max_bound_excess"] == 0.0


def test_calibration_observer_records_independent_layer_kv_amax() -> None:
    pool = PagedKVPool(
        num_layers=2,
        num_pages=1,
        page_size=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.bfloat16,
    )
    observer = gate._KvAmaxObserver(pool)
    observer.install()
    try:
        pool.write_ragged(
            0,
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0]),
            torch.tensor([[[2.0, -3.0]]], dtype=torch.bfloat16),
            torch.tensor([[[4.0, -1.0]]], dtype=torch.bfloat16),
            torch.tensor([0]),
        )
        pool.write_batched(
            1,
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([1]),
            torch.tensor([[[5.0, -2.0]]], dtype=torch.bfloat16),
            torch.tensor([[[1.0, -6.0]]], dtype=torch.bfloat16),
        )
    finally:
        observer.restore()

    assert observer.values() == ((3.0, 5.0), (4.0, 6.0))


def test_checked_in_calibrated_failure_artifact_replays() -> None:
    artifact = (
        Path(__file__).resolve().parents[2]
        / "bench/results/"
        "g4-ekv-fp8-kv-qwen3-32b-sm120-calibrated-fail-2026-08-08"
    )
    verified = gate.verify_artifact(artifact)
    replayed = gate.replay_artifact(artifact)

    assert verified == replayed
    assert verified["verdict"] == "FAIL"
    assert {
        name for name, passed in verified["checks"].items() if not passed
    } == {"all_fp8_writes_audited", "exact_output_tokens_and_stopping"}
