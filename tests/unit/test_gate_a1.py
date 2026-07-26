"""Formal G2 A1 gate assembly must fail closed on incomplete evidence."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from bench.gate_a1 import evaluate

_REPO = Path(__file__).resolve().parents[2]


def _provenance() -> dict:
    return {
        "schema": 3,
        "checkpoint_config_sha256": "a" * 16,
        "checkpoint_weights_sha256": "b" * 16,
        "checkpoint_weight_files": {"model.safetensors": "c" * 64},
        "tokenizer_files_sha256": "d" * 16,
        "tokenizer_vocab_sha256": "e" * 16,
        "prompt_token_ids_sha256": "f" * 16,
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
        "reference": {
            f"p{index}": {
                "prompt": f"prompt {index}",
                "prompt_ids": [index, index + 1],
                "continuation": list(range(16)),
                "hf_top_logprobs": [{"0": -0.1}] * 16,
            }
            for index in range(64)
        },
    }


def _continuations() -> dict:
    outputs = {f"p{index}": list(range(16)) for index in range(64)}
    exact = {
        "prompts": 64,
        "exact_match": 64,
        "match_rate": 1.0,
        "tokens": 1024,
        "token_match_rate": 1.0,
        "median_first_divergence": None,
        "first_mismatch": None,
    }
    return {
        "config": {
            "harness": "real-model ranks on cuda",
            "hardware": {
                "arch": "cuda",
                "device_name": "GPU",
                "device_count": 8,
                "sm": 120,
                "runtime": {
                    "torch_cuda": "12.8",
                    "nccl": "2.27.5",
                    "nvidia_smi": ["0, GPU, 0000:01:00.0, 570.00"],
                },
            },
            "checkpoint": {
                "name": "meta-llama/Llama-3.1-8B-Instruct",
                "architecture": {
                    "model_type": "llama",
                    "hidden_size": 4096,
                    "num_hidden_layers": 32,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "vocab_size": 128256,
                },
                "weights": [
                    {
                        "file": "model.safetensors",
                        "bytes": 1,
                        "sha256": "c" * 64,
                    }
                ],
            },
            "code": {"commit": "1" * 40, "dirty": False},
            "tp_degrees": [1, 2],
            "num_prompts": 64,
            "max_new_tokens": 16,
            "overlap_modes": ["off", "on"],
            "seed": 20260702,
            "sampling": {"method": "greedy"},
        },
        "prompts": [
            {
                "request_id": f"p{index}",
                "text": f"prompt {index}",
                "token_ids": [index, index + 1],
            }
            for index in range(64)
        ],
        "continuations": {
            f"tp{tp}_overlap_{mode}": copy.deepcopy(outputs)
            for tp in (1, 2)
            for mode in ("off", "on")
        },
        "overlap_equality": {
            "tp1_overlap_on_vs_off": copy.deepcopy(exact),
            "tp2_overlap_on_vs_off": copy.deepcopy(exact),
        },
    }


def _teacher(tp: int) -> dict:
    return {
        "config": {
            "reference_provenance": _provenance(),
            "code": {"commit": "1" * 40, "dirty": False},
            "tensor_parallel_size": tp,
            "num_prompts": 64,
            "positions": 16,
            "dtype": "bfloat16",
        },
        "agreement": {"positions": 1024, "verdict": "PASS"},
        "logprob_tolerance": {"verdict": "PASS"},
    }


def _checks(continuations=None, reference=None, teachers=None) -> dict[str, bool]:
    rows = evaluate(
        continuations or _continuations(),
        reference or _reference(),
        teachers or [_teacher(1), _teacher(2)],
    )
    return {row["name"]: row["passed"] for row in rows}


def test_complete_a1_evidence_passes():
    assert all(_checks().values())


def test_missing_raw_continuation_fails():
    evidence = _continuations()
    del evidence["continuations"]["tp2_overlap_on"]["p63"]
    checks = _checks(continuations=evidence)
    assert checks["all raw continuations are retained"] is False


def test_overlap_difference_fails():
    evidence = _continuations()
    evidence["overlap_equality"]["tp2_overlap_on_vs_off"]["exact_match"] = 63
    checks = _checks(continuations=evidence)
    assert checks["overlap ON reproduces OFF at TP1 and TP2"] is False


def test_teacher_forced_failure_fails():
    tp2 = _teacher(2)
    tp2["logprob_tolerance"]["verdict"] = "FAIL"
    checks = _checks(teachers=[_teacher(1), tp2])
    assert checks["amended teacher-forced verdict passes at TP1 and TP2"] is False


def test_cross_checkpoint_evidence_fails():
    reference = _reference()
    reference["provenance"]["checkpoint_weight_files"]["model.safetensors"] = "9" * 64
    teachers = [_teacher(1), _teacher(2)]
    for teacher in teachers:
        teacher["config"]["reference_provenance"] = copy.deepcopy(reference["provenance"])
    checks = _checks(reference=reference, teachers=teachers)
    assert checks["continuation and HF evidence use identical weights"] is False


def test_dirty_or_mixed_commit_evidence_fails():
    tp2 = _teacher(2)
    tp2["config"]["code"]["commit"] = "2" * 40
    checks = _checks(teachers=[_teacher(1), tp2])
    assert checks["all measurements use one clean commit"] is False


def test_gate_command_exits_nonzero_when_an_amended_verdict_fails(tmp_path):
    inputs = {
        "continuations": _continuations(),
        "reference": _reference(),
        "teacher-tp1": _teacher(1),
        "teacher-tp2": _teacher(2),
    }
    inputs["teacher-tp2"]["agreement"]["verdict"] = "FAIL"
    paths = {}
    for name, payload in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        paths[name] = path
    out = tmp_path / "gate.json"

    result = subprocess.run(
        [
            sys.executable,
            "bench/gate_a1.py",
            "--continuations",
            str(paths["continuations"]),
            "--reference",
            str(paths["reference"]),
            "--teacher-tp1",
            str(paths["teacher-tp1"]),
            "--teacher-tp2",
            str(paths["teacher-tp2"]),
            "--out",
            str(out),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    payload = json.loads(out.read_text())
    assert payload["verdict"] == "FAIL"
    failed = {row["name"] for row in payload["checks"] if not row["passed"]}
    assert "amended teacher-forced verdict passes at TP1 and TP2" in failed
