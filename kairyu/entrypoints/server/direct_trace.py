"""Privacy-safe stage traces for direct engine requests (Chat and Responses).

Shared by the Chat Completions handler and the Responses engine path, so an
``X-Kairyu-Trace: 1`` request reports the same ``kairyu_trace`` /
``kairyu_trace_v2`` extensions whichever public API carried it.
"""

from __future__ import annotations

from collections.abc import Sequence

from kairyu.engine.backend import GenerationStageMetric
from kairyu.orchestration.trace import StructuredTrace, TraceEvent, utc_now_iso


def _direct_stage_trace(
    response_id: str,
    *,
    started_at: str,
    stage_metrics: Sequence[GenerationStageMetric],
) -> StructuredTrace:
    """Build the privacy-safe v2 envelope for one direct engine request."""

    stage_metrics = _merge_stage_metrics(stage_metrics)
    events = tuple(
        TraceEvent(
            node=metric.stage,
            kind="stage",
            operation="stage",
            status="success",
            role="engine",
            metadata={
                "stage": metric.stage,
                "duration_ns": metric.duration_ns,
                "occurrences": metric.occurrences,
                "aggregation": "sum",
                "scope": "request-observed",
            },
        )
        for metric in stage_metrics
    )
    return StructuredTrace(
        request_id=response_id,
        started_at=started_at,
        completed_at=utc_now_iso(),
        events=events,
    )


def _merge_stage_metrics(
    *groups: Sequence[GenerationStageMetric],
) -> tuple[GenerationStageMetric, ...]:
    """Combine same-stage observations across nested Kairyu transport hops."""

    order: list[str] = []
    totals: dict[str, tuple[int, int]] = {}
    for group in groups:
        for metric in group:
            if metric.stage not in totals:
                order.append(metric.stage)
                totals[metric.stage] = (0, 0)
            duration_ns, occurrences = totals[metric.stage]
            totals[metric.stage] = (
                duration_ns + metric.duration_ns,
                occurrences + metric.occurrences,
            )
    return tuple(
        GenerationStageMetric(
            stage=stage,
            duration_ns=totals[stage][0],
            occurrences=totals[stage][1],
        )
        for stage in order
    )


def _direct_trace_lines(
    stage_metrics: Sequence[GenerationStageMetric],
) -> list[str]:
    stage_metrics = _merge_stage_metrics(stage_metrics)
    return [
        f"stage:{metric.stage} duration_ns={metric.duration_ns} occurrences={metric.occurrences}"
        for metric in stage_metrics
    ]
