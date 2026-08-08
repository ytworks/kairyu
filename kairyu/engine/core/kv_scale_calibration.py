"""Validated static per-layer K/V scales for experimental FP8 KV caches."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "kairyu.fp8-kv-calibration.v1"
QUANTIZATION = "e4m3-symmetric-static-per-layer-kv"
FP8_MAX = 448.0


def _sha256_json(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _validated_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class KvScaleCalibration:
    checkpoint_sha256: str
    calibration_data_sha256: str
    headroom: float
    k_amax: tuple[float, ...]
    v_amax: tuple[float, ...]
    k_scales: tuple[float, ...]
    v_scales: tuple[float, ...]

    @classmethod
    def from_amax(
        cls,
        *,
        checkpoint_sha256: str,
        calibration_data_sha256: str,
        headroom: float,
        k_amax: tuple[float, ...],
        v_amax: tuple[float, ...],
    ) -> KvScaleCalibration:
        _validated_digest("checkpoint_sha256", checkpoint_sha256)
        _validated_digest("calibration_data_sha256", calibration_data_sha256)
        if not math.isfinite(headroom) or headroom < 1.0:
            raise ValueError("FP8 KV calibration headroom must be finite and >= 1")
        if len(k_amax) == 0 or len(k_amax) != len(v_amax):
            raise ValueError("FP8 KV calibration requires non-empty paired K/V layers")
        for name, values in (("K", k_amax), ("V", v_amax)):
            if any(not math.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError(f"FP8 {name} amax values must be finite and positive")

        def scales(values: tuple[float, ...]) -> tuple[float, ...]:
            return tuple(value * headroom / FP8_MAX for value in values)

        return cls(
            checkpoint_sha256=checkpoint_sha256,
            calibration_data_sha256=calibration_data_sha256,
            headroom=float(headroom),
            k_amax=tuple(float(value) for value in k_amax),
            v_amax=tuple(float(value) for value in v_amax),
            k_scales=scales(k_amax),
            v_scales=scales(v_amax),
        )

    @property
    def num_layers(self) -> int:
        return len(self.k_scales)

    def artifact(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "quantization": QUANTIZATION,
            "fp8_max": FP8_MAX,
            "checkpoint_sha256": self.checkpoint_sha256,
            "calibration_data_sha256": self.calibration_data_sha256,
            "headroom": self.headroom,
            "num_layers": self.num_layers,
            "k_amax": list(self.k_amax),
            "v_amax": list(self.v_amax),
            "k_scales": list(self.k_scales),
            "v_scales": list(self.v_scales),
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.artifact())

    @classmethod
    def from_artifact(
        cls,
        artifact: object,
        *,
        expected_layers: int | None = None,
        expected_checkpoint_sha256: str | None = None,
    ) -> KvScaleCalibration:
        if not isinstance(artifact, dict) or set(artifact) != {
            "schema_version",
            "quantization",
            "fp8_max",
            "checkpoint_sha256",
            "calibration_data_sha256",
            "headroom",
            "num_layers",
            "k_amax",
            "v_amax",
            "k_scales",
            "v_scales",
        }:
            raise ValueError("FP8 KV calibration artifact has an invalid shape")
        if artifact["schema_version"] != SCHEMA_VERSION:
            raise ValueError("FP8 KV calibration artifact has an unsupported schema")
        if artifact["quantization"] != QUANTIZATION or artifact["fp8_max"] != FP8_MAX:
            raise ValueError("FP8 KV calibration artifact has incompatible quantization")
        try:
            k_amax = tuple(float(value) for value in artifact["k_amax"])
            v_amax = tuple(float(value) for value in artifact["v_amax"])
            stored_k = tuple(float(value) for value in artifact["k_scales"])
            stored_v = tuple(float(value) for value in artifact["v_scales"])
        except (TypeError, ValueError) as error:
            raise ValueError("FP8 KV calibration arrays must be numeric") from error
        result = cls.from_amax(
            checkpoint_sha256=_validated_digest(
                "checkpoint_sha256", artifact["checkpoint_sha256"]
            ),
            calibration_data_sha256=_validated_digest(
                "calibration_data_sha256", artifact["calibration_data_sha256"]
            ),
            headroom=float(artifact["headroom"]),
            k_amax=k_amax,
            v_amax=v_amax,
        )
        if artifact["num_layers"] != result.num_layers:
            raise ValueError("FP8 KV calibration layer count does not match its arrays")
        if stored_k != result.k_scales or stored_v != result.v_scales:
            raise ValueError("FP8 KV calibration scales do not match amax/headroom")
        if expected_layers is not None and result.num_layers != expected_layers:
            raise ValueError(
                f"FP8 KV calibration has {result.num_layers} layers; "
                f"expected {expected_layers}"
            )
        if (
            expected_checkpoint_sha256 is not None
            and result.checkpoint_sha256 != expected_checkpoint_sha256
        ):
            raise ValueError("FP8 KV calibration checkpoint identity does not match")
        return result


__all__ = [
    "FP8_MAX",
    "QUANTIZATION",
    "SCHEMA_VERSION",
    "KvScaleCalibration",
]
