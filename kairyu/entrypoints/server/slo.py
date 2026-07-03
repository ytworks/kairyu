"""F5 CPU logic: SLO admission + autoscale decisions (m11 D6/A11).

``AdmissionController`` predicts TTFT from GATEWAY-observable signals only
(in-flight count × EMA of observed TTFT per in-flight unit — engine
internals are invisible through the ZMQ/vLLM backends). Over-SLO requests
shed (429 ``slo_shed``) or defer to batch — the DECISION is recorded; batch
rerouting is the caller's move. ``autoscale_decision`` is a pure function
with hysteresis; execution is a deploy-day HPA/KEDA adapter.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

_EMA_ALPHA = 0.2


@dataclass(frozen=True)
class AdmissionDecision:
    action: str  # "admit" | "shed" | "defer"
    predicted_ttft_s: float
    reason: str = ""


class AdmissionController:
    def __init__(
        self,
        ttft_slo_s: float,
        defer_threshold_s: float | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttft_slo_s <= 0:
            raise ValueError("ttft_slo_s must be > 0")
        self._slo = ttft_slo_s
        # between slo and defer_threshold -> defer to batch; above -> shed
        self._defer_threshold = defer_threshold_s or (ttft_slo_s * 2)
        self._now = now
        self._in_flight = 0
        self._ttft_per_unit_ema = 0.01  # optimistic prior; updated by observe()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def predicted_ttft(self) -> float:
        return (self._in_flight + 1) * self._ttft_per_unit_ema

    def decide(self) -> AdmissionDecision:
        predicted = self.predicted_ttft()
        if predicted <= self._slo:
            return AdmissionDecision("admit", predicted)
        if predicted <= self._defer_threshold:
            return AdmissionDecision(
                "defer", predicted, f"predicted {predicted:.3f}s > SLO {self._slo}s"
            )
        return AdmissionDecision(
            "shed", predicted, f"predicted {predicted:.3f}s > {self._defer_threshold}s"
        )

    def started(self) -> float:
        self._in_flight += 1
        return self._now()

    def finished_first_token(self, started_at: float) -> None:
        """Feed one observed TTFT; per-unit cost = ttft / concurrency then."""
        observed = max(self._now() - started_at, 1e-6)
        per_unit = observed / max(self._in_flight, 1)
        self._ttft_per_unit_ema = (
            (1 - _EMA_ALPHA) * self._ttft_per_unit_ema + _EMA_ALPHA * per_unit
        )

    def completed(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)


@dataclass(frozen=True)
class ScaleDecision:
    action: str  # "scale_up" | "scale_down" | "hold"
    target_delta: int
    reason: str


def autoscale_decision(
    utilization_window: list[float],
    queue_depth: int,
    *,
    high: float = 0.85,
    low: float = 0.3,
    min_window: int = 3,
) -> ScaleDecision:
    """Pure hysteresis policy: sustained high util or queued work scales up;
    sustained low util scales down; anything mixed holds."""
    if len(utilization_window) < min_window:
        return ScaleDecision("hold", 0, "window too short")
    if queue_depth > 0 and min(utilization_window[-min_window:]) >= high:
        return ScaleDecision(
            "scale_up", 1, f"util >= {high} for {min_window} samples with queue"
        )
    if max(utilization_window[-min_window:]) <= low:
        return ScaleDecision("scale_down", -1, f"util <= {low} for {min_window} samples")
    return ScaleDecision("hold", 0, "hysteresis band")
