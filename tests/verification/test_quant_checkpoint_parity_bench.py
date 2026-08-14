from __future__ import annotations

import copy

import pytest

from verification.l1.correctness import quant_checkpoint_parity_bench as gate


def _quantization(scheme: str) -> dict:
    if scheme == "awq":
        return {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "version": "gemm",
            "zero_point": True,
        }
    if scheme == "gptq":
        return {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
        }
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 8,
                    "strategy": "channel",
                    "symmetric": True,
                },
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "token",
                    "dynamic": True,
                    "symmetric": True,
                },
            }
        },
    }


def _payload(scheme: str, reference: dict) -> dict:
    repo, revision = gate.PINS[scheme]
    rows = [
        {
            "prompt": f"p{prompt}",
            "position": position,
            "reference_token": 7,
            "engine_token": 7,
            "reference_top_logprobs": {"7": -0.1, "8": -1.0},
        }
        for prompt in range(gate.PROMPTS)
        for position in range(gate.POSITIONS)
    ]
    return {
        "schema_version": 5,
        "config": {
            "comparison_kind": "quantized-candidate-vs-unquantized-reference",
            "checkpoint_repo": repo,
            "checkpoint_revision": revision,
            "reference_checkpoint_repo": gate.REFERENCE_REPO,
            "reference_checkpoint_revision": gate.REFERENCE_REVISION,
            "reference_provenance": reference,
            "num_prompts": gate.PROMPTS,
            "positions": gate.POSITIONS,
            "tensor_parallel_size": 1,
            "code": {"dirty": False, "commit": "abc"},
            "candidate_provenance": {
                "recorded_top_logprobs": gate.TOP_LOGPROBS,
                "checkpoint_contract": {
                    "quantization_config": _quantization(scheme)
                },
                "checkpoint_weight_files": {"model.safetensors": "a" * 64},
            },
        },
        "raw_positions": rows,
        "logprob_tolerance": {"agreeing_positions_max_abs_delta": 0.5},
        "cross_checkpoint_gate": {
            "expected_positions": gate.EXPECTED_POSITIONS,
            "captured_positions": gate.EXPECTED_POSITIONS,
            "missing_predictions": 0,
            "substantive_disagreements": 0,
            "tie_gap_nats": 0.125,
            "passed": True,
        },
    }


@pytest.mark.parametrize("scheme", ["int8", "awq", "gptq"])
def test_raw_arm_replay_accepts_each_pinned_abi(scheme: str) -> None:
    reference = {"recorded_top_logprobs": gate.TOP_LOGPROBS}
    score = gate._score_arm(scheme, _payload(scheme, reference), reference)
    assert score["agreed"] == gate.EXPECTED_POSITIONS
    assert score["substantive_disagreements"] == 0
    assert score["passed"] is True


def test_raw_arm_replay_recomputes_substantive_disagreement() -> None:
    reference = {"recorded_top_logprobs": gate.TOP_LOGPROBS}
    payload = _payload("awq", reference)
    broken = copy.deepcopy(payload)
    broken["raw_positions"][0]["engine_token"] = 8
    broken["cross_checkpoint_gate"]["substantive_disagreements"] = 1
    broken["cross_checkpoint_gate"]["passed"] = False

    score = gate._score_arm("awq", broken, reference)

    assert score["agreed"] == gate.EXPECTED_POSITIONS - 1
    assert score["substantive_disagreements"] == 1
    assert score["passed"] is False


def test_stored_gate_cannot_disagree_with_raw_replay() -> None:
    reference = {"recorded_top_logprobs": gate.TOP_LOGPROBS}
    payload = _payload("gptq", reference)
    payload["cross_checkpoint_gate"]["passed"] = False

    with pytest.raises(ValueError, match="stored gate differs"):
        gate._score_arm("gptq", payload, reference)
