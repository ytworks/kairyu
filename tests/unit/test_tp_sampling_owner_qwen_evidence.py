"""Pure replay checks for the formal Qwen3-32B ownership artifact (#225)."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch

from bench import tp_sampling_owner_qwen as gate
from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import EngineSampling


def test_python_bin_is_added_to_path_for_flashinfer_jit(monkeypatch) -> None:
    monkeypatch.setattr(gate.sys, "executable", "/venv/bin/python")
    monkeypatch.setattr(gate.shutil, "which", lambda _name, *, path: None)
    monkeypatch.setattr(gate.os, "access", lambda path, mode: path == gate.Path("/venv/bin/ninja"))
    monkeypatch.setenv("PATH", "/usr/bin")

    gate._ensure_python_bin_on_path()

    assert gate.os.environ["PATH"] == "/venv/bin:/usr/bin"


def _record(
    token_id: int,
    logprob: float,
    top: tuple[tuple[int, float], ...],
) -> dict:
    return {
        "token_id": token_id,
        "logprob": logprob,
        "top_logprobs": [
            {"token_id": candidate, "logprob": candidate_logprob}
            for candidate, candidate_logprob in top
        ],
    }


def _diverging_runs() -> tuple[dict, dict, list[dict]]:
    request = {"request_id": "case", "max_new_tokens": 3}
    tp1 = {
        "raw_records": {
            "case": [
                _record(1, -0.10, ((1, -0.10), (2, -0.20))),
                _record(2, -0.20, ((2, -0.20), (3, -0.21))),
                _record(4, -8.00, ((4, -8.00), (5, -9.00))),
            ]
        }
    }
    tp8 = {
        "raw_records": {
            "case": [
                _record(1, -0.12, ((1, -0.12), (2, -0.22))),
                _record(3, -0.22, ((3, -0.22), (2, -0.23))),
                _record(9, -0.01, ((9, -0.01), (8, -0.02))),
            ]
        }
    }
    return tp1, tp8, [request]


def _compatible_probe(position: int = 1) -> list[dict]:
    if position == 0:
        return [
            {
                "request_id": "case",
                "position": 0,
                "tp1_token_id": 1,
                "tp8_token_id": 2,
                "tp1_selected_under_tp1": -0.10,
                "tp1_selected_under_tp8": -0.12,
                "tp8_selected_under_tp1": -0.20,
                "tp8_selected_under_tp8": -0.22,
            }
        ]
    return [
        {
            "request_id": "case",
            "position": 1,
            "tp1_token_id": 2,
            "tp8_token_id": 3,
            "tp1_selected_under_tp1": -0.20,
            "tp1_selected_under_tp8": -0.23,
            "tp8_selected_under_tp1": -0.21,
            "tp8_selected_under_tp8": -0.22,
        }
    ]


def test_free_running_divergence_is_diagnostic_after_compatible_decision():
    tp1, tp8, requests = _diverging_runs()

    checks, summary, diagnostics = gate._compare_runs(
        tp1,
        tp8,
        requests,
        _compatible_probe(),
    )

    assert all(checks.values())
    assert summary["exact_free_running_token_parity_diagnostic"] is False
    assert summary["common_prefix_matching_tokens"] == 1
    assert summary["requests_with_first_divergence"] == 1
    assert summary["post_divergence_distribution_comparison_performed"] is False
    assert any(row["kind"] == "first_free_running_divergence" for row in diagnostics)


def test_first_divergence_uses_direct_probe_not_raw_top64_membership():
    tp1, tp8, requests = _diverging_runs()
    tp8["raw_records"]["case"][1]["top_logprobs"] = [
        {"token_id": 3, "logprob": -0.22},
        {"token_id": 7, "logprob": -0.23},
    ]

    checks, _summary, _diagnostics = gate._compare_runs(
        tp1,
        tp8,
        requests,
        _compatible_probe(),
    )

    assert all(checks.values())


def test_first_divergence_requires_complete_direct_cross_probe():
    tp1, tp8, requests = _diverging_runs()

    checks, _summary, diagnostics = gate._compare_runs(
        tp1,
        tp8,
        requests,
        [],
    )

    assert checks["first_divergence_direct_cross_logprobs_complete"] is False
    assert checks["first_divergence_selected_logprob_tolerance"] is False
    assert any(row["kind"] == "first_divergence_probe_count_failure" for row in diagnostics)


def test_first_divergence_probe_must_match_own_public_logprob():
    tp1, tp8, requests = _diverging_runs()
    probes = _compatible_probe()
    probes[0]["tp1_selected_under_tp1"] = -1.0

    checks, _summary, diagnostics = gate._compare_runs(
        tp1,
        tp8,
        requests,
        probes,
    )

    assert checks["first_divergence_direct_cross_logprobs_complete"] is False
    assert any(
        row["kind"] == "first_divergence_probe_public_logprob_failure" for row in diagnostics
    )


def test_position_zero_divergence_needs_no_hidden_aligned_token():
    tp1, tp8, requests = _diverging_runs()
    tp8["raw_records"]["case"][0]["token_id"] = 2
    tp8["raw_records"]["case"][0]["logprob"] = -0.22

    checks, summary, _diagnostics = gate._compare_runs(
        tp1,
        tp8,
        requests,
        _compatible_probe(position=0),
    )

    assert all(checks.values())
    assert summary["common_prefix_matching_tokens"] == 0


def test_aligned_selected_logprob_drift_remains_binding():
    tp1, tp8, requests = _diverging_runs()
    tp8["raw_records"]["case"][0]["logprob"] = -1.0

    checks, _summary, diagnostics = gate._compare_runs(
        tp1,
        tp8,
        requests,
        _compatible_probe(),
    )

    assert checks["aligned_prefix_selected_logprob_tolerance"] is False
    assert any(row["kind"] == "aligned_selected_logprob_delta_failure" for row in diagnostics)


def test_raw_record_rejects_selected_top64_logprob_contradiction():
    record = {
        "position": 0,
        "token_id": 0,
        "logprob": -1.0,
        "top_logprobs": [
            {"token_id": token_id, "logprob": -float(token_id + 1)}
            for token_id in range(gate.TOP_LOGPROBS)
        ],
        "grammar_terminated": False,
    }
    assert gate._validate_record(record, position=0, vocab_size=1000)

    record["logprob"] = -1.5
    assert not gate._validate_record(record, position=0, vocab_size=1000)


def test_raw_logprob_recorder_retains_full_distribution():
    inner = SimpleNamespace(sampler=Sampler())
    recorded = gate._install_raw_logprob_recorder(inner)
    logits = torch.tensor([3.0, 1.0, -2.0])

    inner.sampler.sample_device(
        "request",
        EngineSampling(temperature=0.0, logprobs=2),
        0,
        logits,
    )

    assert torch.equal(
        recorded["request"][0],
        torch.log_softmax(logits.float(), dim=-1),
    )


def test_sparse_cross_probe_is_extracted_from_full_distributions():
    tp1, tp8, requests = _diverging_runs()
    tp1_distribution = torch.full((10,), -9.0, dtype=torch.float64)
    tp8_distribution = torch.full((10,), -9.0, dtype=torch.float64)
    tp1_distribution[2], tp1_distribution[3] = -0.20, -0.21
    tp8_distribution[2], tp8_distribution[3] = -0.23, -0.22

    probes = gate._build_first_divergence_probes(
        {"tp1": tp1, "tp8": tp8},
        {
            "tp1": {"case": {1: tp1_distribution}},
            "tp8": {"case": {1: tp8_distribution}},
        },
        requests,
    )

    assert probes == _compatible_probe()


def test_source_provenance_must_remain_identical_through_measurement(
    monkeypatch,
):
    monkeypatch.setattr(
        gate,
        "_source_snapshot_is_clean_and_bound",
        lambda _snapshot: True,
    )
    start = {"commit": "a" * 40, "dirty": False}
    source = {
        **start,
        "measurement_end": dict(start),
        "stable_across_measurement": True,
    }
    assert gate._source_is_clean_and_bound(source)

    source["measurement_end"]["dirty"] = True
    assert not gate._source_is_clean_and_bound(source)


def test_fixed_prompt_ids_and_rank_ownership_are_exactly_bound():
    prompt_ids = (
        (785, 6722, 315, 9625, 374),
        (750, 75698, 1445, 982, 262, 421, 308, 2651, 220, 16, 510, 286, 470, 308, 198, 262, 470),
        (785, 2326, 6028, 7987, 525, 2518, 11, 6303, 11, 323),
        (641, 1633, 35085, 15473, 11, 41822, 6147, 264, 536, 311),
        (
            785,
            1156,
            5779,
            10250,
            5109,
            525,
            220,
            17,
            11,
            220,
            18,
            11,
            220,
            20,
            11,
            220,
            22,
            11,
            220,
            16,
            16,
            11,
        ),
        (1249, 3378, 264, 1140, 304, 9931, 1973, 304, 13027, 11, 1494, 279, 5693),
    )
    requests = gate._expected_request_config()
    for request, ids in zip(requests, prompt_ids, strict=True):
        request["prompt_token_ids"] = list(ids)

    assert gate._prompt_token_ids_sha256(requests) == gate.EXPECTED_PROMPT_TOKEN_IDS_SHA256
    topology = {
        "tp_degree": 8,
        "ownership_probe_phase": "after_generation_completed",
        "sampling_ownership": gate._expected_sampling_ownership(8),
    }
    assert gate._sampling_ownership_ok(topology, tp=8)

    forged = deepcopy(topology)
    forged["sampling_ownership"][1]["sampler_present"] = True
    assert not gate._sampling_ownership_ok(forged, tp=8)
