"""CPU contracts and construction plumbing for opt-in FP32 logits (#364)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.config_validation import validate_backend_options
from kairyu.engine.core.logits_dtype import (
    resolve_logits_dtype,
    validate_logits_dtype,
)
from kairyu.models.config import parse_model_config
from kairyu.models.llama import DenseDecoder
from kairyu.models.logits import project_logits

_TINY = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "intermediate_size": 32,
    "vocab_size": 32,
    "max_position_embeddings": 64,
    "tie_word_embeddings": True,
}


class _IdentityComm:
    def tensor_all_reduce(self, tensor):
        return tensor

    def tensor_all_reduce_max(self, tensor):
        return tensor


@pytest.mark.parametrize("value", ("model", "float32"))
def test_public_logits_dtype_names(value) -> None:
    assert validate_logits_dtype(value) == value


@pytest.mark.parametrize(
    "value",
    (None, "", "bfloat16", "fp32", torch.float32, 1, True),
)
def test_public_logits_dtype_rejects_unknown_values(value) -> None:
    with pytest.raises(ValueError, match="logits_dtype must be one of"):
        validate_logits_dtype(value)


@pytest.mark.parametrize(
    ("requested", "model_dtype", "resolved"),
    (
        ("model", torch.float16, "float16"),
        ("model", torch.bfloat16, "bfloat16"),
        ("model", torch.float32, "float32"),
        ("float32", torch.bfloat16, "float32"),
    ),
)
def test_logits_dtype_resolution(requested, model_dtype, resolved) -> None:
    assert resolve_logits_dtype(requested, model_dtype) == resolved


def test_model_policy_is_the_exact_existing_head_call() -> None:
    torch.manual_seed(364)
    head = torch.nn.Linear(8, 5, bias=True)
    hidden = torch.randn(2, 3, 8)

    expected = head(hidden)
    actual = project_logits(head, hidden, "model")

    assert torch.equal(actual, expected)


def test_float32_policy_accepts_an_already_float32_cpu_model() -> None:
    torch.manual_seed(365)
    head = torch.nn.Linear(8, 5, bias=True)
    hidden = torch.randn(2, 3, 8)

    actual = project_logits(head, hidden, "float32")
    expected = torch.nn.functional.linear(hidden, head.weight, head.bias)

    assert actual.dtype is torch.float32
    assert actual.shape == (2, 3, 5)
    torch.testing.assert_close(actual, expected)


def test_float32_policy_fails_closed_for_a_non_dense_or_cpu_bf16_head() -> None:
    hidden = torch.randn(2, 8)
    with pytest.raises(RuntimeError, match="dense torch.nn.Linear"):
        project_logits(torch.nn.Identity(), hidden, "float32")

    bf16_head = torch.nn.Linear(8, 5, bias=False).to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="supports CUDA.*float32"):
        project_logits(bf16_head, hidden.to(torch.bfloat16), "float32")


def test_dense_decoder_validates_at_construction_and_preserves_tied_weight() -> None:
    config = parse_model_config(_TINY)
    with pytest.raises(ValueError, match="logits_dtype must be one of"):
        DenseDecoder(config, logits_dtype="bf16")

    decoder = DenseDecoder(config, logits_dtype="float32")
    assert decoder.logits_dtype == "float32"
    assert decoder.lm_head.weight is decoder.model.embed_tokens.weight
    before = tuple(decoder.state_dict())
    output = decoder.logits(torch.randn(4, config.hidden_size))
    assert output.dtype is torch.float32
    assert decoder.lm_head.weight is decoder.model.embed_tokens.weight
    assert tuple(decoder.state_dict()) == before


def test_dense_decoder_rejects_invalid_logits_dtype_before_backbone_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kairyu.models import llama

    def unexpected_backbone(*_args, **_kwargs):
        pytest.fail("backbone was constructed before logits_dtype validation")

    monkeypatch.setattr(llama, "_Backbone", unexpected_backbone)
    with pytest.raises(ValueError, match="logits_dtype must be one of"):
        DenseDecoder(parse_model_config(_TINY), logits_dtype="bf16")


def test_real_model_builders_reject_invalid_logits_dtype_before_io_or_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kairyu.engine.core import hw_profile
    from kairyu.engine.core.pd_factory import build_pd_coordinator
    from kairyu.models.loader import load_model
    from kairyu.models.moe_parallel import build_ep_model
    from kairyu.models.parallel import build_tp_model

    monkeypatch.setattr(
        hw_profile,
        "probe",
        lambda *_args, **_kwargs: pytest.fail(
            "hardware was probed before logits_dtype validation"
        ),
    )
    calls = (
        lambda: load_model("/does-not-exist", logits_dtype="bf16"),
        lambda: build_tp_model(
            "/does-not-exist",
            tp=2,
            rank=0,
            comm=None,
            logits_dtype="bf16",
        ),
        lambda: build_ep_model(
            "/does-not-exist",
            ep_size=2,
            ep_rank=0,
            comm=None,
            logits_dtype="bf16",
        ),
        lambda: build_pd_coordinator(
            model_path="/does-not-exist",
            logits_dtype="bf16",
        ),
    )
    for call in calls:
        with pytest.raises(ValueError, match="logits_dtype must be one of"):
            call()


@pytest.fixture
def tiny_checkpoint(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = transformers.LlamaForCausalLM(config).to(torch.float32).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    return tmp_path


def test_loader_and_tp_builder_propagate_logits_dtype(tiny_checkpoint) -> None:
    from kairyu.models.loader import load_model
    from kairyu.models.parallel import build_tp_model

    loaded, _config, _generation = load_model(
        tiny_checkpoint,
        logits_dtype="float32",
    )
    sharded, _local, _full = build_tp_model(
        str(tiny_checkpoint),
        tp=1,
        rank=0,
        comm=_IdentityComm(),
        logits_dtype="float32",
    )

    assert loaded.logits_dtype == "float32"
    assert sharded.logits_dtype == "float32"
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert sharded.lm_head.weight is sharded.model.embed_tokens.weight


def test_pipeline_stage_uses_the_full_model_logits_policy() -> None:
    from kairyu.engine.core.pp_worker import PpStageModel

    decoder = DenseDecoder(parse_model_config(_TINY), logits_dtype="float32")
    stage = PpStageModel(decoder, num_stages=1, stage=0)
    hidden = torch.randn(3, _TINY["hidden_size"])

    assert stage.logits_dtype == "float32"
    torch.testing.assert_close(stage.logits(hidden), decoder.logits(hidden))


@pytest.mark.parametrize("backend", ("kairyu", "kairyu-proc"))
@pytest.mark.parametrize("value", ("model", "float32"))
def test_native_config_accepts_logits_dtype_for_a_real_model(backend, value) -> None:
    validate_backend_options(
        backend,
        {"model_path": "/model", "logits_dtype": value},
    )


@pytest.mark.parametrize("backend", ("kairyu", "kairyu-proc"))
def test_native_config_rejects_float32_logits_without_a_real_model(backend) -> None:
    with pytest.raises(ValueError, match="requires a real model_path"):
        validate_backend_options(backend, {"logits_dtype": "float32"})


def test_vllm_config_rejects_the_native_logits_dtype_option() -> None:
    with pytest.raises(ValueError, match="does not support.*logits_dtype"):
        validate_backend_options(
            "vllm",
            {"model": "/model", "logits_dtype": "float32"},
        )


def test_benchmark_attestation_fails_closed_on_missing_or_ignored_mode() -> None:
    from bench.parity_tp import _logits_dtype_identity

    with pytest.raises(RuntimeError, match="request mismatch"):
        _logits_dtype_identity(SimpleNamespace(), "model")
    with pytest.raises(RuntimeError, match="silently ignored"):
        _logits_dtype_identity(
            SimpleNamespace(
                logits_dtype_requested="float32",
                logits_dtype_resolved="bfloat16",
            ),
            "float32",
        )
    assert _logits_dtype_identity(
        SimpleNamespace(
            logits_dtype_requested="float32",
            logits_dtype_resolved="float32",
        ),
        "float32",
    ) == {"requested": "float32", "resolved": "float32"}


def test_parity_hf_cli_forwards_and_records_the_constructed_float32_arm(
    tmp_path,
    monkeypatch,
) -> None:
    from bench import parity_hf
    from kairyu.engine.core import hw_profile
    from kairyu.engine.core.sampling_types import SampledToken

    provenance = {"num_prompts": 1, "positions": 1}
    reference = {
        "p0": {
            "prompt": "The",
            "prompt_ids": [1],
            "continuation": [7],
            "hf_top_logprobs": [{"7": -0.1, "8": -1.0}],
        }
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps({"provenance": provenance, "reference": reference})
    )
    output_path = tmp_path / "float32.json"
    seen = {}

    def fake_engine(*args, **kwargs):
        seen["logits_dtype"] = kwargs.get("logits_dtype")
        return (
            {
                "p0": [
                    SampledToken(
                        7,
                        logprob=-0.1,
                        top_logprobs=((7, -0.1), (8, -1.0)),
                    )
                ]
            },
            {"requested": "float32", "resolved": "float32"},
        )

    monkeypatch.setattr(parity_hf, "_provenance", lambda *args, **kwargs: provenance)
    monkeypatch.setattr(parity_hf, "_engine_next_tokens", fake_engine)
    monkeypatch.setattr(
        parity_hf,
        "_code_provenance",
        lambda: {"commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        hw_profile,
        "probe",
        lambda: SimpleNamespace(
            arch="cpu",
            device_name=None,
            device_count=0,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_hf.py",
            "--model-path",
            "/unused",
            "--num-prompts",
            "1",
            "--positions",
            "1",
            "--logits-dtype",
            "float32",
            "--reference",
            str(reference_path),
            "--out",
            str(output_path),
        ],
    )

    assert parity_hf.main() == 0
    config = json.loads(output_path.read_text())["config"]
    assert seen["logits_dtype"] == "float32"
    assert config["logits_dtype_requested"] == "float32"
    assert config["logits_dtype_resolved"] == "float32"


def test_paired_position_summary_counts_moves_in_both_directions() -> None:
    from bench.gate_logits_dtype import paired_position_summary

    def row(position, reference, engine, delta):
        return {
            "prompt": "p0",
            "position": position,
            "reference_token": reference,
            "engine_token": engine,
            "reference_token_logprob": -0.5,
            "engine_token_logprob": (
                -0.5 - delta if delta is not None else -0.5
            ),
            "absolute_logprob_delta": delta,
        }

    shared_contract = {
        "reference_noise_floor": {"raw_self_agreement_rate": 0.98},
        "logprob_tolerance": {"tie_gap_nats": 0.125, "top_k": 20},
    }
    model = {
        **shared_contract,
        "raw_positions": [
            row(0, 10, 11, None),
            row(1, 20, 20, 0.10),
            row(2, 30, 31, None),
        ]
    }
    float32 = {
        **shared_contract,
        "raw_positions": [
            row(0, 10, 10, 0.05),
            row(1, 20, 21, None),
            row(2, 30, 32, None),
        ]
    }

    summary = paired_position_summary(model, float32)

    assert summary["complete"] is True
    assert summary["reference_contract_match"] is True
    assert summary["reference_self_agreement_floor"] == 0.98
    assert summary["tie_gap_nats"] == 0.125
    assert summary["model_agreed"] == summary["float32_agreed"] == 1
    assert summary["agreement_delta"] == 0
    assert summary["classification"] == "same"
    assert summary["toward_reference"] == 1
    assert summary["away_from_reference"] == 1
    assert summary["changed_other"] == 1
    assert summary["agreeing_position_logprob_delta"]["model"]["max"] == pytest.approx(
        0.10
    )
    assert summary["agreeing_position_logprob_delta"]["float32"][
        "max"
    ] == pytest.approx(0.05)


def test_paired_assembler_rejects_missing_raw_evidence_and_mixed_arm_labels() -> None:
    from bench.gate_logits_dtype import evaluate

    def artifact(gate, arm):
        resolved = "float32" if arm == "float32" else "bfloat16"
        degrees = (1, 2) if gate == "A1" else (2, 4, 8)
        evidence = {
            "reference": {"reference_runtime": {"code": {}}},
            **{
                f"teacher_tp{tp}": {
                    "config": {
                        "logits_dtype_requested": arm,
                        "logits_dtype_resolved": resolved,
                    }
                }
                for tp in degrees
            },
        }
        if gate == "A1":
            evidence["continuations"] = {
                "config": {
                    "logits_dtype_requested": arm,
                    "logits_dtype_resolved": resolved,
                }
            }
        return {"gate": f"G2 {gate}", "verdict": "PASS", "evidence": evidence}

    a1_model = artifact("A1", "model")
    # One TP row that claims the control arm inside the treatment artifact must
    # fail even though the outer gate label says float32 and PASS.
    a1_float32 = artifact("A1", "float32")
    a1_float32["evidence"]["teacher_tp2"]["config"][
        "logits_dtype_requested"
    ] = "model"

    checks, _summaries, _quality = evaluate(
        a1_model,
        a1_float32,
        artifact("A2", "model"),
        artifact("A2", "float32"),
    )
    by_name = {check["name"]: check for check in checks}

    assert by_name[
        "A1 float32 uses one attested logits mode at every observation"
    ]["passed"] is False
    assert by_name["A1 TP1 has complete paired raw positions"]["passed"] is False
    assert by_name["A2 TP8 has complete paired raw positions"]["passed"] is False
