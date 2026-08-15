from __future__ import annotations

import pytest

from verification.l1.performance import serving_bench


def _execution_trace(execution_status: object) -> dict:
    return {
        "trace_version": "2.0",
        "request_id": "req-1",
        "started_at": "2026-08-15T00:00:00.000Z",
        "completed_at": "2026-08-15T00:00:01.000Z",
        "events": [
            {
                "seq": 1,
                "node": "exec_matrix",
                "role": "executor",
                "kind": "execution",
                "status": "success",
                "attempt": 0,
                "timing": {
                    "queued_at": "2026-08-15T00:00:00.100Z",
                    "started_at": "2026-08-15T00:00:00.100Z",
                    "first_token_at": None,
                    "completed_at": "2026-08-15T00:00:00.200Z",
                },
                "usage": None,
                "detail": {"execution_status": execution_status},
            }
        ],
    }


def test_parse_trace_retains_execution_status_in_artifact_stage() -> None:
    version, stages = serving_bench._parse_trace(
        _execution_trace("unavailable"),
        response_id="req-1",
    )

    assert version == "2.0"
    assert len(stages) == 1
    assert stages[0].execution_status == "unavailable"
    assert stages[0].as_dict()["execution_status"] == "unavailable"


@pytest.mark.parametrize(
    "execution_status",
    [None, "UNAVAILABLE", "ok,", "ok,unavailable,", "x" * 129],
)
def test_parse_trace_rejects_unsafe_execution_status(
    execution_status: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="execution trace must report a safe execution_status",
    ):
        serving_bench._parse_trace(
            _execution_trace(execution_status),
            response_id="req-1",
        )
