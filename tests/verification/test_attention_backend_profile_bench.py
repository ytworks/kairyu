"""CPU-only verification tests for the #277 attention evidence artifact."""

import copy
import json

import pytest

from verification.l1.performance import attention_backend_profile_bench as attention_bench


def _measurements(*, fa4_faster: bool, clock_base: int) -> list[dict]:
    measurements = []
    clock = clock_base
    for block in range(attention_bench.PAIRED_BLOCKS):
        pair_order, backend_order = attention_bench._expected_order(block)
        durations = (
            {
                attention_bench.FLASHINFER: 2_000 + block,
                attention_bench.FLASHATTENTION4: 1_000 + block,
            }
            if fa4_faster
            else {
                attention_bench.FLASHINFER: 1_000 + block,
                attention_bench.FLASHATTENTION4: 2_000 + block,
            }
        )
        for position, backend in enumerate(backend_order):
            started = clock
            finished = started + durations[backend]
            measurements.append(
                {
                    "block": block,
                    "pair_order": pair_order,
                    "position": position,
                    "backend": backend,
                    "started_ns": started,
                    "finished_ns": finished,
                    "elapsed_ns": finished - started,
                }
            )
            clock = finished + 100
    return measurements


def _parity(output_shape: list[int]) -> dict:
    return {
        "reference": attention_bench.FLASHINFER,
        "candidate": attention_bench.FLASHATTENTION4,
        "shape": output_shape,
        "candidate_shape": output_shape,
        "rtol": attention_bench.PARITY_RTOL,
        "atol": attention_bench.PARITY_ATOL,
        "max_abs_error": 0.001,
        "max_normalized_error": 0.05,
        "reference_sha256": "1" * 64,
        "candidate_sha256": "2" * 64,
        "passed": True,
    }


def _valid_artifact(*, fa4_faster: bool = True) -> dict:
    scenarios = []
    for index, scenario in enumerate(attention_bench.SCENARIOS):
        geometry = attention_bench._scenario_geometry(scenario)
        measurements = _measurements(
            fa4_faster=fa4_faster,
            clock_base=1_000_000_000 * (index + 1),
        )
        scenarios.append(
            {
                "name": scenario.name,
                "geometry": geometry,
                "parity": _parity(geometry["output_shape"]),
                "measurements": measurements,
                "statistics": attention_bench.compute_statistics(measurements),
            }
        )
    environment = {
        "device": "cuda",
        "packages": {
            "torch": "2.test",
            "flashinfer": "0.test",
            "flash-attn-4": None,
        },
        "cuda": {
            "torch_cuda_version": "13.test",
            "cudnn_version": None,
        },
        "gpu": {
            "name": "test SM120",
            "compute_capability": [12, 0],
            "sm": 120,
            "total_memory_bytes": 96_000_000_000,
        },
    }
    provenance = {
        "generated_at_utc": "2026-07-29T00:00:00+00:00",
        "source": {
            "commit": "a" * 40,
            "dirty": False,
            "status_sha256": "b" * 64,
            "script_sha256": attention_bench._script_sha256(),
        },
        "python_version": "3.12.test",
        "platform": "test-platform",
    }
    return attention_bench.assemble_artifact(
        scenarios,
        environment=environment,
        provenance=provenance,
    )


def test_nearest_rank_and_exact_sign_test_are_transparent() -> None:
    assert (
        attention_bench.nearest_rank_percentile(list(range(1, 101)), 0.99)
        == 99
    )
    pairs = [(2_000 + index, 1_000 + index) for index in range(24)]
    result = attention_bench.exact_fa4_faster_sign_test(pairs)
    assert result["wins"] == 24
    assert result["losses"] == result["ties"] == 0
    assert result["tail_numerator"] == 1
    assert result["tail_denominator"] == 2**24
    assert result["significant"] is True


def test_valid_artifact_reconstructs_fa4_recommendation() -> None:
    artifact = _valid_artifact()
    report = attention_bench.verify_artifact(artifact)

    assert report["valid"] is True
    assert all(report["checks"].values())
    assert report["recomputed_recommendation"] == artifact["recommendation"]
    assert artifact["recommendation"]["prefill_backend"] == "flashattention4"
    assert artifact["recommendation"]["decode_backend"] == "flashinfer"
    assert artifact["recommendation"]["minimum_effect_size_threshold"] is None


def _mutate_median(artifact: dict) -> None:
    artifact["scenarios"][0]["statistics"]["latency_ns"]["flashinfer"][
        "median"
    ] += 1


def _mutate_order(artifact: dict) -> None:
    artifact["scenarios"][0]["measurements"][0]["pair_order"] = "BA"


def _mutate_elapsed(artifact: dict) -> None:
    artifact["scenarios"][0]["measurements"][0]["elapsed_ns"] += 1


def _mutate_geometry(artifact: dict) -> None:
    artifact["scenarios"][0]["geometry"]["num_query_heads"] += 1


def _mutate_recommendation(artifact: dict) -> None:
    artifact["recommendation"]["prefill_backend"] = "flashinfer"


def _mutate_parity(artifact: dict) -> None:
    artifact["scenarios"][0]["parity"]["passed"] = False


def _mutate_script_provenance(artifact: dict) -> None:
    artifact["provenance"]["source"]["script_sha256"] = "f" * 64


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_median,
        _mutate_order,
        _mutate_elapsed,
        _mutate_geometry,
        _mutate_recommendation,
        _mutate_parity,
        _mutate_script_provenance,
    ],
)
def test_resealed_semantic_forgery_is_rejected(mutation) -> None:
    artifact = copy.deepcopy(_valid_artifact())
    mutation(artifact)
    forged = attention_bench.seal_artifact(artifact)

    assert attention_bench.verify_artifact(forged)["valid"] is False


def test_unsealed_mutation_is_rejected_by_artifact_hash() -> None:
    artifact = _valid_artifact()
    artifact["environment"]["gpu"]["name"] = "forged"

    report = attention_bench.verify_artifact(artifact)

    assert report["valid"] is False
    assert report["checks"]["artifact_hash"] is False


def test_assert_policy_compares_reconstructed_recommendation(monkeypatch) -> None:
    artifact = _valid_artifact()
    monkeypatch.setattr(
        attention_bench,
        "_current_auto_components",
        lambda: {
            "requested": "auto",
            "source": "hw_profile",
            "prefill": "flashinfer",
            "decode": "flashinfer",
        },
    )

    report = attention_bench.verify_artifact(artifact, assert_policy=True)

    assert report["valid"] is False
    assert report["checks"]["selector_auto_policy_matches"] is False


def test_stable_fallback_artifact_matches_flashinfer_policy(monkeypatch) -> None:
    artifact = _valid_artifact(fa4_faster=False)
    monkeypatch.setattr(
        attention_bench,
        "_current_auto_components",
        lambda: {
            "requested": "auto",
            "source": "hw_profile",
            "prefill": "flashinfer",
            "decode": "flashinfer",
        },
    )

    report = attention_bench.verify_artifact(artifact, assert_policy=True)

    assert artifact["recommendation"]["prefill_backend"] == "flashinfer"
    assert report["valid"] is True


def test_verify_cli_accepts_valid_artifact(tmp_path, capsys) -> None:
    path = tmp_path / "attention.json"
    path.write_text(json.dumps(_valid_artifact()), encoding="utf-8")

    exit_code = attention_bench.main(
        ["verify", "--artifact", str(path)]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
