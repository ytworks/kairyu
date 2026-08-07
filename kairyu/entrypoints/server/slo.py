"""F5 CPU logic: SLO admission + autoscale decisions (m11 D6/A11).

``AdmissionController`` predicts TTFT from GATEWAY-observable signals only:
known ingress elapsed time plus interactive in-flight count × an EMA of
post-admission TTFT per admission-time interactive concurrency unit. Engine
internals are invisible through the ZMQ/vLLM backends. Total active work
separately bounds the deferred backlog. Over-SLO requests shed or defer to
lower-priority batch scheduling. ``begin`` atomically makes that decision and
reserves an in-flight slot for admitted/deferred work; the caller remains
responsible for applying the returned action before dispatch.
``autoscale_decision`` is a pure function with hysteresis; execution is a
deploy-day HPA/KEDA adapter.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

_EMA_ALPHA = 0.2


@dataclass(frozen=True)
class AdmissionDecision:
    action: str  # "admit" | "shed" | "defer"
    predicted_ttft_s: float
    reason: str = ""


@dataclass(frozen=True)
class AdmissionSnapshot:
    in_flight: int
    interactive_in_flight: int
    deferred_in_flight: int
    ttft_per_unit_ema_s: float
    predicted_ttft_s: float
    defer_pressure_s: float


class AdmissionLease:
    """One atomic SLO decision and, unless shed, one in-flight reservation."""

    def __init__(
        self,
        controller: AdmissionController,
        decision: AdmissionDecision,
        *,
        started_at: float | None,
        concurrency_at_start: int | None,
    ) -> None:
        self.decision = decision
        self.started_at = started_at
        self.concurrency_at_start = concurrency_at_start
        self._controller = controller
        self._first_token_observed = False
        self._completed = decision.action == "shed"
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        with self._lock:
            return not self._completed

    def finished_first_token(self) -> None:
        """Record TTFT exactly once for admitted or deferred work."""

        with self._lock:
            if self._completed:
                raise RuntimeError("cannot observe first token after completion")
            if self._first_token_observed:
                raise RuntimeError("first token already observed")
            if self.started_at is None or self.concurrency_at_start is None:
                raise AssertionError("active admission lease has no start state")
            self._controller._finished_first_token(
                self.started_at,
                self.concurrency_at_start,
                observe=self.decision.action == "admit",
            )
            self._first_token_observed = True

    def completed(self) -> None:
        """Release the reserved in-flight slot exactly once."""

        with self._lock:
            if self._completed:
                raise RuntimeError("admission lease already completed")
            self._controller._completed(self.decision.action)
            self._completed = True


class AdmissionController:
    def __init__(
        self,
        ttft_slo_s: float,
        defer_threshold_s: float | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(ttft_slo_s) or ttft_slo_s <= 0:
            raise ValueError("ttft_slo_s must be finite and > 0")
        if defer_threshold_s is None:
            defer_threshold_s = ttft_slo_s * 2
        if (
            not math.isfinite(defer_threshold_s)
            or defer_threshold_s <= ttft_slo_s
        ):
            raise ValueError(
                "defer_threshold_s must be finite and > ttft_slo_s"
            )
        self._slo = ttft_slo_s
        # between slo and defer_threshold -> defer to batch; above -> shed
        self._defer_threshold = defer_threshold_s
        self._now = now
        self._in_flight = 0
        self._interactive_in_flight = 0
        self._deferred_in_flight = 0
        self._ttft_per_unit_ema = 0.01  # optimistic prior; updated by observe()
        self._lock = threading.Lock()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def _predicted_ttft_unlocked(self, elapsed_s: float = 0.0) -> float:
        return elapsed_s + (
            (self._interactive_in_flight + 1) * self._ttft_per_unit_ema
        )

    def _defer_pressure_unlocked(self, elapsed_s: float = 0.0) -> float:
        return elapsed_s + ((self._in_flight + 1) * self._ttft_per_unit_ema)

    @staticmethod
    def _validate_elapsed(elapsed_s: float) -> None:
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and >= 0")

    def predicted_ttft(self, elapsed_s: float = 0.0) -> float:
        self._validate_elapsed(elapsed_s)
        with self._lock:
            return self._predicted_ttft_unlocked(elapsed_s)

    def batch_should_yield(self) -> bool:
        """Whether new batch work should wait for interactive pressure to fall."""

        with self._lock:
            return (
                self._interactive_in_flight > 0
                and self._predicted_ttft_unlocked() > self._slo
            )

    def _decide_unlocked(self, elapsed_s: float = 0.0) -> AdmissionDecision:
        predicted = self._predicted_ttft_unlocked(elapsed_s)
        if predicted <= self._slo:
            return AdmissionDecision("admit", predicted)
        defer_pressure = self._defer_pressure_unlocked(elapsed_s)
        if defer_pressure <= self._defer_threshold:
            return AdmissionDecision(
                "defer", predicted, f"predicted {predicted:.3f}s > SLO {self._slo}s"
            )
        return AdmissionDecision(
            "shed",
            predicted,
            (
                f"defer pressure {defer_pressure:.3f}s "
                f"> {self._defer_threshold}s"
            ),
        )

    def decide(self, elapsed_s: float = 0.0) -> AdmissionDecision:
        """Return a diagnostic decision without reserving capacity.

        Production callers should use :meth:`begin` so another caller cannot
        enter between the decision and the in-flight increment.
        """

        self._validate_elapsed(elapsed_s)
        with self._lock:
            return self._decide_unlocked(elapsed_s)

    def begin(self, elapsed_s: float = 0.0) -> AdmissionLease:
        """Atomically decide and reserve, including known ingress elapsed time."""

        self._validate_elapsed(elapsed_s)
        with self._lock:
            decision = self._decide_unlocked(elapsed_s)
            if decision.action == "shed":
                return AdmissionLease(
                    self,
                    decision,
                    started_at=None,
                    concurrency_at_start=None,
                )
            self._in_flight += 1
            if decision.action == "admit":
                self._interactive_in_flight += 1
                concurrency_at_start = self._interactive_in_flight
            else:
                self._deferred_in_flight += 1
                concurrency_at_start = self._in_flight
            return AdmissionLease(
                self,
                decision,
                started_at=self._now(),
                concurrency_at_start=concurrency_at_start,
            )

    def snapshot(self) -> AdmissionSnapshot:
        """Return bounded raw predictor state for metrics and evidence."""

        with self._lock:
            return AdmissionSnapshot(
                in_flight=self._in_flight,
                interactive_in_flight=self._interactive_in_flight,
                deferred_in_flight=self._deferred_in_flight,
                ttft_per_unit_ema_s=self._ttft_per_unit_ema,
                predicted_ttft_s=self._predicted_ttft_unlocked(),
                defer_pressure_s=self._defer_pressure_unlocked(),
            )

    def started(self) -> float:
        """Compatibility path; new integrations should use :meth:`begin`."""

        with self._lock:
            self._in_flight += 1
            self._interactive_in_flight += 1
            return self._now()

    def finished_first_token(self, started_at: float) -> None:
        """Compatibility path using observation-time concurrency."""

        with self._lock:
            concurrency = self._in_flight
        self._finished_first_token(started_at, concurrency, observe=True)

    def _finished_first_token(
        self,
        started_at: float,
        concurrency_at_start: int,
        *,
        observe: bool,
    ) -> None:
        """Feed one observed TTFT with concurrency frozen at admission."""

        if not math.isfinite(started_at):
            raise ValueError("started_at must be finite")
        if concurrency_at_start < 1:
            raise ValueError("concurrency_at_start must be >= 1")
        observed = max(self._now() - started_at, 1e-6)
        if not math.isfinite(observed):
            raise ValueError("observed TTFT must be finite")
        if not observe:
            return
        per_unit = observed / concurrency_at_start
        with self._lock:
            self._ttft_per_unit_ema = (
                (1 - _EMA_ALPHA) * self._ttft_per_unit_ema
                + _EMA_ALPHA * per_unit
            )

    def completed(self) -> None:
        """Compatibility path; release one started request."""

        self._completed("admit")

    def _completed(self, action: str) -> None:
        with self._lock:
            if self._in_flight < 1:
                raise RuntimeError("cannot complete with no in-flight request")
            if action == "admit":
                if self._interactive_in_flight < 1:
                    raise RuntimeError(
                        "cannot complete with no interactive in-flight request"
                    )
                self._interactive_in_flight -= 1
            elif action == "defer":
                if self._deferred_in_flight < 1:
                    raise RuntimeError(
                        "cannot complete with no deferred in-flight request"
                    )
                self._deferred_in_flight -= 1
            else:
                raise ValueError(f"cannot complete admission action {action!r}")
            self._in_flight -= 1


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
