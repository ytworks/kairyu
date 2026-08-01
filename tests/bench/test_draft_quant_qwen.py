from __future__ import annotations

import pytest

from bench import draft_quant_qwen as gate


def _row(*, accepted: int, draft_ms: float, verify_ms: float) -> dict[str, object]:
    return {
        "accepted": accepted,
        "proposed": 3,
        "committed": accepted + 1,
        "draft_ms": draft_ms,
        "verify_ms": verify_ms,
        "cycle_ms": draft_ms + verify_ms,
        "target_logits_finite": True,
        "target_correction_valid": True,
    }


def test_accepted_prefix_stops_at_the_first_target_mismatch():
    assert gate.accepted_prefix([1, 2, 3], [1, 2, 3]) == 3
    assert gate.accepted_prefix([1, 9, 3], [1, 2, 3]) == 1
    assert gate.accepted_prefix([9, 2, 3], [1, 2, 3]) == 0
    with pytest.raises(ValueError, match="widths differ"):
        gate.accepted_prefix([1], [1, 2])


def test_eagle_context_drops_first_token_and_appends_target_root():
    sequence = (10, 11, 12, 13, 14)
    assert gate.eagle_shifted_context(sequence, 3) == (11, 12, 13)
    assert gate.eagle_shifted_context(sequence, 3) != sequence[:3]
    with pytest.raises(ValueError, match="context and a target root"):
        gate.eagle_shifted_context(sequence, 0)


def test_eagle_verification_includes_root_and_scores_rejection_or_bonus():
    sequence = (10, 11, 12, 13, 14, 15, 16, 17)
    tokens, positions = gate.eagle_verification_geometry(
        sequence,
        anchor=3,
        proposals=[14, 99, 16],
    )
    assert tokens == [13, 14, 99, 16]
    assert positions == (3, 4, 5, 6)

    assert gate.score_eagle_verification(
        sequence,
        anchor=3,
        proposals=[14, 99, 16],
        target_tokens=[14, 15, 16],
        bonus_token=999,
    ) == (1, 15, True)
    assert gate.score_eagle_verification(
        sequence,
        anchor=3,
        proposals=[14, 15, 16],
        target_tokens=[14, 15, 16],
        bonus_token=17,
    ) == (3, 17, True)

    # An unshifted draft commonly proposes the already-produced root first.
    # Root-prefixed target verification correctly rejects it at position zero.
    assert gate.score_eagle_verification(
        sequence,
        anchor=3,
        proposals=[13, 14, 15],
        target_tokens=[14, 15, 16],
        bonus_token=17,
    ) == (0, 14, True)


def test_fp8_metadata_declares_the_validated_dialect_and_dense_exclusions():
    config = gate.fp8_config(("fc", "lm_head"))
    group = config["config_groups"]["group_0"]
    assert group["targets"] == ["Linear"]
    assert group["weights"] == {
        "type": "float",
        "num_bits": 8,
        "strategy": "channel",
        "symmetric": True,
    }
    assert group["input_activations"]["dynamic"] is True
    assert group["input_activations"]["strategy"] == "token"
    assert config["ignore"] == ["fc", "lm_head"]


def test_arm_grid_covers_fc_and_lm_head_dense_quantized_combinations():
    assert {spec.ignored_layers for spec in gate.ARMS} == {
        None,
        (),
        ("fc",),
        ("lm_head",),
        ("fc", "lm_head"),
    }


def test_summary_uses_committed_tokens_and_full_cycle_wall_time():
    summary = gate.summarize_arm(
        "fp8",
        module_bytes=400,
        cuda_bytes=420,
        forward_peak_extra_bytes=32,
        draft_logits_finite=True,
        draft_tokens=3,
        rows=[
            _row(accepted=2, draft_ms=2.0, verify_ms=3.0),
            _row(accepted=1, draft_ms=4.0, verify_ms=6.0),
        ],
    )
    assert summary["acceptance_rate"] == pytest.approx(3 / 6)
    assert summary["acceptance_by_position"] == [1.0, 0.5, 0.0]
    assert summary["committed"] == 5
    assert summary["cycle_committed_tokens_per_second"] == pytest.approx(5 / 0.015)
    assert summary["draft_latency_ms"]["median"] == 3.0
    assert summary["cycle_latency_ms"]["p95"] == 10.0
    assert summary["forward_peak_extra_bytes"] == 32
    assert summary["draft_logits_finite"] is True
    assert summary["target_logits_finite"] is True
    assert summary["target_corrections_valid"] is True


def test_assert_gate_writes_failed_artifact_before_nonzero_exit(tmp_path, monkeypatch):
    artifact = {"gate": {"passed": False, "reason": "measured slowdown"}}
    monkeypatch.setattr(gate, "run", lambda _args: artifact)
    output = tmp_path / "failed.json"

    status = gate.main(
        [
            "--model-path",
            str(tmp_path),
            "--draft-path",
            str(tmp_path),
            "--output",
            str(output),
            "--assert-gate",
        ]
    )

    assert status == 1
    assert output.is_file()
    assert '"passed": false' in output.read_text(encoding="utf-8")
