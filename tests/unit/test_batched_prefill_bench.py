"""The #224 formal artifact gate rejects incomplete structural evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).parents[2] / "bench" / "batched_prefill_qwen.py"
_SPEC = importlib.util.spec_from_file_location("batched_prefill_qwen", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def test_formal_source_contract_binds_shared_profiler_dependency():
    assert "kairyu/bench/profiling.py" in bench.IMPLEMENTATION_FILES


def _stats(rank: int, enabled: bool):
    requests, layers = 8, 64
    return {
        "rank": rank,
        "world_size": 8,
        "device": f"cuda:{rank}",
        "stats": {
            "enabled": enabled,
            "capability_gap": None,
            "rows": requests,
            "model_calls": 1 if enabled else requests,
            "batched_groups": 1 if enabled else 0,
            "sequential_rows": 0 if enabled else requests,
            "backend": [
                {
                    "type": "FlashInferBackend",
                    "plans": 1 if enabled else requests,
                    "runs": layers if enabled else requests * layers,
                }
            ],
        },
    }


def _artifact():
    baseline_schedule = [
        {
            "request_id": f"baseline-{index}",
            "num_tokens": length,
            "is_prefill": True,
        }
        for index, length in enumerate(bench.PROMPT_LENGTHS)
    ]
    candidate_schedule = [
        {**row, "request_id": row["request_id"].replace("baseline", "candidate")}
        for row in baseline_schedule
    ]
    topology = [
        {
            "rank": rank,
            "control_world_size": 8,
            "control_backend": "gloo",
            "model_world_size": 8,
            "model_backend": "nccl",
            "sampling_owner": rank == 0,
            "sampler_present": rank == 0,
            "device": f"cuda:{rank}",
        }
        for rank in range(8)
    ]
    source_snapshot = {
        "commit": "c" * 40,
        "tracked_dirty": False,
        "files": {name: "a" * 64 for name in bench.IMPLEMENTATION_FILES},
        "files_match_commit": True,
    }
    checkpoint_snapshot = {
        "layout": {
            "architecture": ["Qwen3ForCausalLM"],
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
        },
        "config_sha256": bench._EXPECTED_CONFIG_SHA256,
        "index_sha256": bench._EXPECTED_INDEX_SHA256,
        "weights": [dict(row) for row in bench._EXPECTED_WEIGHT_ROWS],
        "weights_total_bytes": 65_524_328_560,
        "weights_manifest_sha256": (bench._EXPECTED_WEIGHTS_MANIFEST_SHA256),
    }
    prompts = []
    for index, length in enumerate(bench.PROMPT_LENGTHS):
        tokens = bench._prompt_tokens(index, length, 151936)
        prompts.append(
            {
                "length": length,
                "first": list(tokens[:4]),
                "last": list(tokens[-4:]),
                "sha256": hashlib.sha256(
                    json.dumps(tokens, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    return {
        "schema_version": bench.SCHEMA_VERSION,
        "measurement_kind": bench.MEASUREMENT_KIND,
        "config": {
            "tp": 8,
            "num_pages": 4096,
            "page_size": 16,
            "prompt_lengths": list(bench.PROMPT_LENGTHS),
            "max_new_tokens": 1,
            "timing_is_diagnostic": True,
        },
        "source": {
            "measurement_start": source_snapshot,
            "measurement_end": source_snapshot,
            "stable_across_measurement": True,
        },
        "checkpoint": {
            "measurement_start": checkpoint_snapshot,
            "measurement_end": checkpoint_snapshot,
            "stable_across_measurement": True,
        },
        "hardware": {
            "devices": [
                {
                    "index": rank,
                    "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                    "compute_capability": "12.0",
                    "uuid": f"GPU-{rank:02d}",
                    "pci": f"0000:{rank:02x}:00",
                    "total_memory_bytes": 100_000_000_000,
                }
                for rank in range(8)
            ]
        },
        "prompts": prompts,
        "topology": topology,
        "baseline": {
            "schedule": baseline_schedule,
            "outputs": list(range(8)),
            "rank_stats": [_stats(rank, False) for rank in range(8)],
            "rank0_cuda_events": 800,
        },
        "candidate": {
            "schedule": candidate_schedule,
            "outputs": list(range(8)),
            "rank_stats": [_stats(rank, True) for rank in range(8)],
            "rank0_cuda_events": 200,
        },
    }


def test_formal_gate_accepts_complete_exact_structural_evidence(monkeypatch):
    monkeypatch.setattr(bench, "_source_gate", lambda source: True)
    checks = bench._evaluate(_artifact())
    assert checks
    assert all(checks.values())


def test_formal_gate_rejects_missing_rank_and_non_reduced_events(
    monkeypatch,
):
    monkeypatch.setattr(bench, "_source_gate", lambda source: True)
    artifact = _artifact()
    artifact["candidate"]["rank_stats"].pop()
    artifact["candidate"]["rank0_cuda_events"] = 900

    checks = bench._evaluate(artifact)

    assert checks["all_rank_candidate_counts"] is False
    assert checks["rank0_cuda_events_reduced"] is False


def test_formal_gate_rejects_uncommitted_source_and_wrong_checkpoint():
    artifact = _artifact()

    checks = bench._evaluate(artifact)
    assert checks["clean_committed_source_stable"] is False

    artifact["checkpoint"]["measurement_start"]["weights"][0]["size"] += 1
    artifact["checkpoint"]["measurement_end"] = artifact["checkpoint"]["measurement_start"]
    assert bench._checkpoint_gate(artifact["checkpoint"]) is False


def test_formal_gate_rejects_implausible_or_duplicate_gpu_identity():
    artifact = _artifact()
    artifact["hardware"]["devices"][0]["total_memory_bytes"] = 1
    artifact["hardware"]["devices"][1]["uuid"] = artifact["hardware"]["devices"][0]["uuid"]

    assert bench._hardware_gate(artifact["hardware"]) is False


def test_verify_rejects_tampered_stored_verdict(monkeypatch, tmp_path: Path):
    artifact = _artifact()
    monkeypatch.setattr(bench, "_source_gate", lambda source: True)
    monkeypatch.setattr(bench, "_hardware", lambda: artifact["hardware"])
    checkpoint = artifact["checkpoint"]["measurement_start"]
    monkeypatch.setattr(bench, "_checkpoint_manifest", lambda path: checkpoint)
    artifact["checks"] = bench._evaluate(artifact)
    artifact["passed"] = all(artifact["checks"].values())
    artifact["checks"]["eight_first_tokens_exact"] = False
    artifact["passed"] = False
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact))

    result = bench._verify(path, tmp_path)

    assert result["checks"]["stored_replay_matches"] is False
    assert result["passed"] is False
