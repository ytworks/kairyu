"""The #215 formal artifact rejects incomplete or timing-based evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).parents[2] / "bench" / "batched_spec_verify_qwen.py"
_SPEC = importlib.util.spec_from_file_location("batched_spec_verify_qwen", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


class _FixtureTokenizer:
    def __init__(self) -> None:
        self.fragments = {
            text: (100 + index * 10, 101 + index * 10, 102 + index * 10, 103 + index * 10)
            for index, (_category, _name, text) in enumerate(bench.PROMPT_SPECS)
        }

    def encode(self, text: str) -> tuple[int, ...]:
        return self.fragments[text]


def _checkpoint_snapshot():
    return {
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
        "weights_manifest_sha256": bench._EXPECTED_WEIGHTS_MANIFEST_SHA256,
    }


def _topology():
    return [
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


def _hardware():
    return {
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
    }


def _rank_stats(*, enabled: bool, graph_runner: bool, structural: bool):
    requests = bench.REQUEST_COUNT
    positions = requests * (bench.SPECULATIVE_TOKENS + 1)
    rows = []
    for rank in range(8):
        graph = None
        if graph_runner:
            graph = {
                "graph_executions": (
                    (1 if enabled else 0) if structural else (5 if enabled else 4)
                ),
                "eager_fallbacks": 0,
                "captured_buckets": [8, positions],
            }
        plans, runs = (
            (1, 0)
            if structural and enabled and graph_runner
            else (1, 64)
            if structural and enabled
            else (positions, positions * 64)
            if structural
            else (7, 7 * 64)
        )
        rows.append(
            {
                "rank": rank,
                "world_size": 8,
                "device": f"cuda:{rank}",
                "stats": {
                    "enabled": enabled,
                    "capability_gap": None,
                    "requests": requests,
                    "positions": positions,
                    "model_calls": 1 if enabled else positions,
                    "batched_groups": 1 if enabled else 0,
                    "sequential_positions": 0 if enabled else positions,
                    "graph_groups": 1 if enabled and graph_runner else 0,
                    "graph": graph,
                    "backend": [
                        {
                            "type": "FlashInferBackend",
                            "plans": plans,
                            "runs": runs,
                        }
                    ],
                },
            }
        )
    return rows


def _structural_cell(*, prefix: str, enabled: bool, graph_runner: bool):
    return {
        "enabled": enabled,
        "schedule": [
            {
                "request_id": f"{prefix}-{index}",
                "position": 1,
                "num_tokens": bench.SPECULATIVE_TOKENS + 1,
                "is_prefill": False,
            }
            for index in range(bench.REQUEST_COUNT)
        ],
        "drafts": [
            [200 + index, 201 + index, 202 + index]
            for index in range(bench.REQUEST_COUNT)
        ],
        "targets": [
            [300 + index, 301 + index, 302 + index, 303 + index]
            for index in range(bench.REQUEST_COUNT)
        ],
        "rank_stats": _rank_stats(
            enabled=enabled,
            graph_runner=graph_runner,
            structural=True,
        ),
    }


def _timing_pair(*, graph_runner: bool):
    outputs = [
        [1000 + request * 10 + position for position in range(bench.MAX_NEW_TOKENS)]
        for request in range(bench.REQUEST_COUNT)
    ]
    samples = []
    orders = [
        list(bench._order_for_round(index))
        for index in range(bench.FORMAL_ROUNDS)
    ]
    for round_index, order in enumerate(orders):
        for slot, mode in enumerate(order):
            enabled = mode == "candidate"
            wall = 1.0 + round_index * 0.1 + slot * 0.01
            samples.append(
                {
                    "round": round_index,
                    "slot": slot,
                    "mode": mode,
                    "outputs": outputs,
                    "draft_proposed": 24,
                    "draft_accepted": 12,
                    "rank_stats": _rank_stats(
                        enabled=enabled,
                        graph_runner=graph_runner,
                        structural=False,
                    ),
                    "timing": {
                        "binding": False,
                        "wall_s": wall,
                        "tpot_s": wall / 64,
                        "tokens_per_s": 64 / wall,
                    },
                }
            )
    return {
        "orders": orders,
        "samples": samples,
        "diagnostic": {
            "binding": False,
            "summary": bench._diagnostic_summary(samples),
        },
    }


def _microcomparison(prompts):
    pages, _num_pages = bench._micro_page_geometry(prompts)
    geometry = []
    selected = []
    for index, (prompt, page_table) in enumerate(
        zip(prompts, pages, strict=True)
    ):
        prompt_len = len(prompt["prompt_token_ids"])
        inputs = [
            prompt["draft_probe_committed"],
            *prompt["expected_ngram_draft"],
        ]
        geometry.append(
            {
                "request_index": index,
                "prompt_length": prompt_len,
                "page_table": list(page_table),
                "target_input_ids": inputs,
                "positions": list(range(prompt_len, prompt_len + 4)),
                "seq_lens": list(range(prompt_len + 1, prompt_len + 5)),
                "write_from": prompt_len,
            }
        )
        selected.append([500 + index * 4 + offset for offset in range(4)])
    orders = []
    samples = []
    for round_index in range(bench.FORMAL_ROUNDS):
        order = (
            [
                "flattened_decode",
                "native_ragged_prefill",
                "native_ragged_prefill",
                "flattened_decode",
            ]
            if round_index % 2 == 0
            else [
                "native_ragged_prefill",
                "flattened_decode",
                "flattened_decode",
                "native_ragged_prefill",
            ]
        )
        orders.append(order)
        for slot, mode in enumerate(order):
            samples.append(
                {
                    "round": round_index,
                    "slot": slot,
                    "mode": mode,
                    "wall_s": 0.1 + round_index * 0.01 + slot * 0.001,
                    "cuda_ms": 90.0 + round_index + slot * 0.1,
                    "selected_token_sha256": "d" * 64,
                }
            )
    return {
        "available": True,
        "device": "cuda:0",
        "checkpoint_layout": {
            "architecture": "Qwen3ForCausalLM",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
        },
        "geometry": geometry,
        "parity": {
            "binding": True,
            "flattened_selected_tokens": selected,
            "native_ragged_selected_tokens": [list(row) for row in selected],
            "selected_tokens_equal": True,
            "flattened_kv_sha256": "e" * 64,
            "native_ragged_kv_sha256": "e" * 64,
            "kv_digest_match_diagnostic": True,
            "flattened_kv_shape": {
                "k": [64, 32, 8, 128],
                "v": [64, 32, 8, 128],
            },
            "native_ragged_kv_shape": {
                "k": [64, 32, 8, 128],
                "v": [64, 32, 8, 128],
            },
            "kv_shapes_equal": True,
            "flattened_target_slots_finite": True,
            "native_ragged_target_slots_finite": True,
            "kv_allclose_2e2_diagnostic": False,
            "kv_max_abs_diff_diagnostic": 5.875,
            "kv_nrmse_diagnostic": 0.012,
            "kv_cosine_similarity_diagnostic": 0.999,
        },
        "timing": {
            "binding": False,
            "orders": orders,
            "samples": samples,
            "summary": bench._micro_summary(samples),
        },
    }


def _artifact():
    checkpoint = _checkpoint_snapshot()
    source_snapshot = {
        "commit": "c" * 40,
        "tracked_dirty": False,
        "files": {name: "a" * 64 for name in bench.IMPLEMENTATION_FILES},
        "files_match_commit": True,
    }
    eager = {
        "timing": _timing_pair(graph_runner=False),
        "structural": {
            "sequential": _structural_cell(
                prefix="structural-eager-sequential",
                enabled=False,
                graph_runner=False,
            ),
            "candidate": _structural_cell(
                prefix="structural-eager-candidate",
                enabled=True,
                graph_runner=False,
            ),
        },
    }
    graph = {
        "timing": _timing_pair(graph_runner=True),
        "structural": {
            "sequential": _structural_cell(
                prefix="structural-cuda-graph-sequential",
                enabled=False,
                graph_runner=True,
            ),
            "candidate": _structural_cell(
                prefix="structural-cuda-graph-candidate",
                enabled=True,
                graph_runner=True,
            ),
        },
    }
    prompts = bench._prompt_rows(_FixtureTokenizer())
    return {
        "schema_version": bench.SCHEMA_VERSION,
        "measurement_kind": bench.MEASUREMENT_KIND,
        "config": {
            "tp": 8,
            "num_pages": bench.NUM_PAGES,
            "page_size": bench.PAGE_SIZE,
            "request_count": bench.REQUEST_COUNT,
            "speculative_tokens": bench.SPECULATIVE_TOKENS,
            "max_new_tokens": bench.MAX_NEW_TOKENS,
            "pattern_repeats": bench.PATTERN_REPEATS,
            "rounds": bench.FORMAL_ROUNDS,
            "graph_max_batch": bench.GRAPH_MAX_BATCH,
            "graph_max_pages": bench.GRAPH_MAX_PAGES,
            "timing_is_diagnostic": True,
        },
        "gate_policy": {
            "timing_binding": False,
            "binding_basis": [
                "source_checkpoint_hardware_provenance",
                "all_rank_model_call_and_position_counts",
                "zero_cuda_graph_fallback",
                "end_to_end_and_target_token_parity",
                "single_gpu_flattened_vs_native_ragged_token_and_kv_write_coverage",
            ],
            "diagnostic_only": [
                "wall_s",
                "tpot_s",
                "tokens_per_s",
                "cuda_ms",
                "median",
                "mad",
                "kv_digest_match_diagnostic",
                "kv_allclose_2e2_diagnostic",
                "kv_max_abs_diff_diagnostic",
                "kv_nrmse_diagnostic",
                "kv_cosine_similarity_diagnostic",
            ],
        },
        "source": {
            "measurement_start": source_snapshot,
            "measurement_end": source_snapshot,
            "stable_across_measurement": True,
        },
        "checkpoint": {
            "measurement_start": checkpoint,
            "measurement_end": checkpoint,
            "stable_across_measurement": True,
        },
        "hardware": _hardware(),
        "prompts": prompts,
        "topology": {
            "eager": _topology(),
            "cuda_graph": _topology(),
        },
        "measurements": {
            "eager": eager,
            "cuda_graph": graph,
        },
        "native_ragged_microcomparison": _microcomparison(prompts),
    }


def test_gate_accepts_complete_structural_parity_evidence(monkeypatch):
    monkeypatch.setattr(bench, "_source_gate", lambda source: True)

    checks = bench._evaluate(_artifact())

    assert checks
    assert all(checks.values())


def test_gate_rejects_missing_rank_and_wrong_call_count(monkeypatch):
    monkeypatch.setattr(bench, "_source_gate", lambda source: True)
    artifact = _artifact()
    structural = artifact["measurements"]["eager"]["structural"]["candidate"]
    structural["rank_stats"].pop()

    checks = bench._evaluate(artifact)

    assert checks["batched_eager_counts"] is False

    artifact = _artifact()
    artifact["measurements"]["eager"]["structural"]["candidate"]["rank_stats"][0][
        "stats"
    ]["model_calls"] = 2
    checks = bench._evaluate(artifact)
    assert checks["batched_eager_counts"] is False


def test_gate_rejects_cuda_graph_fallback(monkeypatch):
    monkeypatch.setattr(bench, "_source_gate", lambda source: True)
    artifact = _artifact()
    graph = artifact["measurements"]["cuda_graph"]
    graph["structural"]["candidate"]["rank_stats"][3]["stats"]["graph"][
        "eager_fallbacks"
    ] = 1

    checks = bench._evaluate(artifact)

    assert checks["batched_graph_counts_without_fallback"] is False


def test_gate_rejects_source_and_checkpoint_drift():
    artifact = _artifact()

    checks = bench._evaluate(artifact)
    assert checks["clean_committed_source_stable"] is False

    artifact["checkpoint"]["measurement_start"]["weights"][0]["size"] += 1
    artifact["checkpoint"]["measurement_end"] = artifact["checkpoint"][
        "measurement_start"
    ]
    assert bench._checkpoint_gate(artifact["checkpoint"]) is False


def test_gate_rejects_timing_marked_binding(monkeypatch):
    monkeypatch.setattr(bench, "_source_gate", lambda source: True)
    artifact = _artifact()
    artifact["gate_policy"]["timing_binding"] = True
    artifact["measurements"]["eager"]["timing"]["diagnostic"]["binding"] = True
    artifact["measurements"]["eager"]["timing"]["samples"][0]["timing"][
        "binding"
    ] = True

    checks = bench._evaluate(artifact)

    assert checks["timing_strictly_non_binding"] is False
    assert checks["eager_abba_baab_complete"] is False


def test_gate_rejects_native_ragged_token_or_missing_kv_write(monkeypatch):
    monkeypatch.setattr(bench, "_source_gate", lambda source: True)
    artifact = _artifact()
    micro = artifact["native_ragged_microcomparison"]
    micro["parity"]["native_ragged_selected_tokens"][0][0] += 1

    checks = bench._evaluate(artifact)

    assert checks["native_ragged_microcomparison_exact"] is False

    artifact = _artifact()
    artifact["native_ragged_microcomparison"]["parity"][
        "native_ragged_target_slots_finite"
    ] = False
    checks = bench._evaluate(artifact)
    assert checks["native_ragged_microcomparison_exact"] is False


def test_verify_rejects_tampered_stored_verdict(monkeypatch, tmp_path: Path):
    artifact = _artifact()
    monkeypatch.setattr(bench, "_source_gate", lambda source: True)
    monkeypatch.setattr(bench, "_hardware", lambda: artifact["hardware"])
    checkpoint = artifact["checkpoint"]["measurement_start"]
    monkeypatch.setattr(bench, "_checkpoint_manifest", lambda path: checkpoint)
    monkeypatch.setattr(bench, "_prompt_tokenizer_replay", lambda rows, path: True)
    artifact["checks"] = bench._evaluate(artifact)
    artifact["passed"] = all(artifact["checks"].values())
    artifact["checks"]["end_to_end_and_target_parity"] = False
    artifact["passed"] = False
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact))

    result = bench._verify(path, tmp_path)

    assert result["checks"]["stored_replay_matches"] is False
    assert result["passed"] is False
