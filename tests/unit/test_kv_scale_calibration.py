import copy

import pytest

from kairyu.engine.core.kv_scale_calibration import KvScaleCalibration


def _calibration() -> KvScaleCalibration:
    return KvScaleCalibration.from_amax(
        checkpoint_sha256="a" * 64,
        calibration_data_sha256="b" * 64,
        headroom=1.05,
        k_amax=(4.0, 8.0),
        v_amax=(2.0, 6.0),
    )


def test_calibration_derives_per_layer_kv_scales_and_round_trips():
    calibration = _calibration()

    assert calibration.k_scales == (4.0 * 1.05 / 448.0, 8.0 * 1.05 / 448.0)
    assert calibration.v_scales == (2.0 * 1.05 / 448.0, 6.0 * 1.05 / 448.0)
    assert len(calibration.sha256) == 64
    assert (
        KvScaleCalibration.from_artifact(
            calibration.artifact(),
            expected_layers=2,
            expected_checkpoint_sha256="a" * 64,
        )
        == calibration
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("k_scales", [1.0, 1.0], "do not match"),
        ("num_layers", 3, "layer count"),
        ("checkpoint_sha256", "wrong", "SHA-256"),
    ],
)
def test_calibration_rejects_tampered_artifacts(field, value, match):
    artifact = copy.deepcopy(_calibration().artifact())
    artifact[field] = value

    with pytest.raises(ValueError, match=match):
        KvScaleCalibration.from_artifact(artifact)


def test_calibration_rejects_wrong_checkpoint_and_geometry():
    artifact = _calibration().artifact()

    with pytest.raises(ValueError, match="expected 3"):
        KvScaleCalibration.from_artifact(artifact, expected_layers=3)
    with pytest.raises(ValueError, match="checkpoint identity"):
        KvScaleCalibration.from_artifact(
            artifact, expected_checkpoint_sha256="c" * 64
        )
