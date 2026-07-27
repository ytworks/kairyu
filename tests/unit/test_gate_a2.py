"""Formal G2 A2 assembly fails closed on partial or mixed 70B evidence."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from bench.gate_a2 import evaluate

_REPO = Path(__file__).resolve().parents[2]
_REVISION = "a" * 40


def _provenance() -> dict:
    return {
        "schema": 5,
        "checkpoint_config_sha256": "b" * 16,
        "checkpoint_contract": {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {"type": "float", "num_bits": 8},
                        "input_activations": {
                            "type": "float",
                            "num_bits": 8,
                            "dynamic": True,
                        },
                    }
                },
            },
        },
        "checkpoint_weights_sha256": "c" * 16,
        "checkpoint_weight_files": {
            f"model-{index:05d}-of-00015.safetensors": "d" * 64
            for index in range(1, 16)
        },
        "tokenizer_files_sha256": "e" * 16,
        "tokenizer_vocab_sha256": "f" * 16,
        "prompt_token_ids_sha256": "1" * 16,
        "num_prompts": 64,
        "positions": 16,
        "generation": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        },
    }


def _reference() -> dict:
    return {
        "provenance": _provenance(),
        "reference_runtime": {
            "code": {"commit": "2" * 40, "dirty": False},
            "checkpoint_repo": "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
            "checkpoint_revision": _REVISION,
        },
        "reference": {
            f"p{prompt}": {
                "prompt": f"prompt {prompt}",
                "prompt_ids": [prompt],
                "continuation": list(range(16)),
                "hf_top_logprobs": [
                    {str(position): -0.1, "999": -3.0}
                    for position in range(16)
                ],
            }
            for prompt in range(64)
        },
    }


def _teacher(tp: int) -> dict:
    raw = [
        {
            "prompt": f"p{prompt}",
            "position": position,
            "reference_token": position,
            "engine_token": position,
            "agreement": True,
            "reference_token_logprob": -0.1,
            "engine_token_logprob": -0.1,
            "absolute_logprob_delta": 0.0,
            "reference_top_logprobs": {str(position): -0.1, "999": -3.0},
        }
        for prompt in range(64)
        for position in range(16)
    ]
    return {
        "config": {
            "checkpoint_repo": "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
            "checkpoint_revision": _REVISION,
            "reference_provenance": _provenance(),
            "code": {"commit": "2" * 40, "dirty": False},
            "tensor_parallel_size": tp,
            "num_prompts": 64,
            "positions": 16,
            "dtype": "bfloat16",
            "hardware": {
                "arch": "cuda",
                "runtime": {
                    "nccl": "2.27.5",
                    "nvidia_smi": ["GPU"],
                    "nvidia_smi_topology": ["GPU0 GPU1"],
                },
            },
        },
        "reference_noise_floor": {
            "raw_self_agreement_rate": 1.0,
        },
        "agreement": {"positions": 1024, "verdict": "PASS"},
        "logprob_tolerance": {
            "substantive": 0,
            "missing_samples": [],
            "verdict": "PASS",
        },
        "raw_positions": raw,
    }


def _checks(reference=None, teachers=None):
    rows, comparisons = evaluate(
        reference or _reference(),
        teachers or [_teacher(2), _teacher(4), _teacher(8)],
    )
    return {row["name"]: row["passed"] for row in rows}, comparisons


def test_complete_a2_evidence_passes():
    checks, comparisons = _checks()
    assert all(checks.values())
    assert all(row["verdict"] == "PASS" for row in comparisons.values())


def test_missing_raw_position_fails():
    tp4 = _teacher(4)
    tp4["raw_positions"].pop()
    checks, _ = _checks(teachers=[_teacher(2), tp4, _teacher(8)])
    assert checks["TP2, TP4, and TP8 retain every raw position"] is False


def test_wrong_model_contract_fails():
    reference = _reference()
    reference["provenance"]["checkpoint_contract"]["hidden_size"] = 4096
    teachers = [_teacher(tp) for tp in (2, 4, 8)]
    for teacher in teachers:
        teacher["config"]["reference_provenance"] = copy.deepcopy(
            reference["provenance"]
        )
    checks, _ = _checks(reference=reference, teachers=teachers)
    assert checks["checkpoint is Llama-3.3-70B FP8 W8A8"] is False


def test_substantive_tp4_disagreement_against_tp2_fails():
    tp4 = _teacher(4)
    tp4["raw_positions"][0]["engine_token"] = 12345
    tp4["raw_positions"][0]["agreement"] = False
    checks, comparisons = _checks(
        teachers=[_teacher(2), tp4, _teacher(8)]
    )
    assert comparisons["tp4_vs_tp2"]["substantive"] == 1
    assert checks["TP4 and TP8 pass directly against TP2"] is False


def test_reported_pass_cannot_hide_raw_hf_disagreements():
    teachers = [_teacher(tp) for tp in (2, 4, 8)]
    for teacher in teachers:
        teacher["raw_positions"][0]["engine_token"] = 12345
        teacher["raw_positions"][0]["agreement"] = False
    checks, comparisons = _checks(teachers=teachers)
    assert all(
        teacher["agreement"]["verdict"] == "PASS" for teacher in teachers
    )
    assert comparisons["tp2_vs_hf"]["verdict"] == "FAIL"
    assert checks["raw rows recompute both amended criteria at every TP degree"] is False


def test_distribution_delta_is_retained_but_is_not_a_third_a2_criterion():
    tp2 = _teacher(2)
    tp2["raw_positions"][0]["engine_token_logprob"] = -0.5
    tp2["logprob_tolerance"]["verdict"] = "FAIL"
    checks, comparisons = _checks(
        teachers=[tp2, _teacher(4), _teacher(8)]
    )
    score = comparisons["tp2_vs_hf"]
    assert score["max_abs_logprob_delta"] == 0.4
    assert score["distribution_delta_verdict"] == "DIAGNOSTIC_EXCEEDANCE"
    assert score["verdict"] == "PASS"
    assert checks["every TP degree passes the amended HF-relative criteria"] is True


def test_dirty_or_mixed_measurement_commit_fails():
    tp8 = _teacher(8)
    tp8["config"]["code"]["dirty"] = True
    checks, _ = _checks(teachers=[_teacher(2), _teacher(4), tp8])
    assert checks["all measurements use one clean commit"] is False


def test_gate_command_exits_nonzero_on_failed_amended_verdict(tmp_path):
    inputs = {
        "reference": _reference(),
        "teacher-tp2": _teacher(2),
        "teacher-tp4": _teacher(4),
        "teacher-tp8": _teacher(8),
    }
    inputs["teacher-tp8"]["agreement"]["verdict"] = "FAIL"
    paths = {}
    for name, payload in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        paths[name] = path
    out = tmp_path / "gate.json"
    result = subprocess.run(
        [
            sys.executable,
            "bench/gate_a2.py",
            "--reference",
            str(paths["reference"]),
            "--teacher-tp2",
            str(paths["teacher-tp2"]),
            "--teacher-tp4",
            str(paths["teacher-tp4"]),
            "--teacher-tp8",
            str(paths["teacher-tp8"]),
            "--out",
            str(out),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert json.loads(out.read_text())["verdict"] == "FAIL"
