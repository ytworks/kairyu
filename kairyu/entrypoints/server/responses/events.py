"""Responses SSE event framing and data-heartbeat cadence for every stream path."""

from __future__ import annotations

from kairyu.entrypoints.server.stream_util import sse_event

# Generation can stay silent for minutes (prefill, thinking, orchestrated
# stages) while Codex retries any SSE stream without a *data* event for 300s;
# comment lines never reset its idle timer. Every stream path therefore
# repeats response.in_progress after this much data silence. Stream paths read
# it as ``events.HEARTBEAT_SECONDS`` at call time, so one value governs them all.
HEARTBEAT_SECONDS = 15.0


def responses_sse(event_type: str, sequence_number: int, **payload) -> str:
    """One Responses stream event with its gapless sequence number."""

    return sse_event(
        event_type,
        {"type": event_type, "sequence_number": sequence_number, **payload},
    )
