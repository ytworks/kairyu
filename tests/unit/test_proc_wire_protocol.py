"""Versioned process-wire encoding and reconstruction tests."""

from __future__ import annotations

import pytest

from kairyu.engine.core.engine_service import (
    LEGACY_WIRE_VERSION,
    WIRE_VERSION,
    WireEventCursor,
    event_from_update,
)
from kairyu.engine.engine_loop import StreamUpdate
from kairyu.engine.zmq_backend import EngineServiceError, _WireAccumulator
from kairyu.outputs import TokenLogprob


def _content(token_id: int, token: str) -> TokenLogprob:
    return TokenLogprob(
        token=token,
        token_id=token_id,
        logprob=-0.25,
        bytes_=tuple(token.encode()),
        top=(
            TokenLogprob(
                token="?",
                token_id=token_id + 10,
                logprob=-1.25,
                bytes_=(63,),
            ),
        ),
    )


def _update(
    outputs: tuple[int, ...],
    text: str,
    *,
    finished: bool = False,
    logprobs: tuple[dict[int, float], ...] | None = None,
    content: tuple[TokenLogprob, ...] | None = None,
) -> StreamUpdate:
    return StreamUpdate(
        outputs=outputs,
        text=text,
        finished=finished,
        finish_reason="length" if finished else None,
        logprobs=logprobs,
        cumulative_logprob=-0.25 * len(outputs),
        num_prompt_tokens=7,
        num_cached_tokens=4,
        logprob_content=content,
    )


def _encode(cursor: WireEventCursor, update: StreamUpdate) -> dict:
    event = event_from_update(
        "request",
        update,
        wire_version=WIRE_VERSION,
        cursor=cursor,
    )
    assert event is not None
    event["stream_id"] = "stream"
    return event


def test_v2_snapshot_and_deltas_reconstruct_every_cumulative_field():
    cursor = WireEventCursor()
    accumulator = _WireAccumulator()
    updates = (
        _update(
            (1,),
            "caf",
            logprobs=({1: -0.25, 11: -1.25},),
            content=(_content(1, "c"),),
        ),
        _update(
            (1, 2),
            "café",
            logprobs=(
                {1: -0.25, 11: -1.25},
                {2: -0.25, 12: -1.25},
            ),
            content=(_content(1, "c"), _content(2, "é")),
        ),
        # Final exact detokenization rewrites the unstable tail ("é" -> "e!").
        _update(
            (1, 2, 3),
            "cafe!",
            finished=True,
            logprobs=(
                {1: -0.25, 11: -1.25},
                {2: -0.25, 12: -1.25},
                {3: -0.25, 13: -1.25},
            ),
            content=(
                _content(1, "c"),
                _content(2, "é"),
                _content(3, "!"),
            ),
        ),
    )

    events = [_encode(cursor, update) for update in updates]
    assert events[0]["event"] == "snapshot"
    assert events[1]["event"] == events[2]["event"] == "delta"
    assert events[2]["text_offset"] == 3
    for event in events[1:]:
        assert {"outputs", "text", "logprobs", "logprob_content"}.isdisjoint(event)

    normalized = None
    for event in events:
        normalized = accumulator.apply(event)
    assert normalized is not None
    assert normalized["outputs"] == [1, 2, 3]
    assert normalized["text"] == "cafe!"
    assert normalized["finished"] is True
    assert normalized["finish_reason"] == "length"
    assert normalized["num_prompt_tokens"] == 7
    assert normalized["num_cached_tokens"] == 4
    assert normalized["cumulative_logprob"] == -0.75
    assert len(normalized["logprobs"]) == 3
    assert len(normalized["logprob_content"]) == 3


def test_v2_suppresses_empty_nonterminal_updates_but_keeps_terminal():
    cursor = WireEventCursor()
    update = _update((), "")
    snapshot = event_from_update(
        "request",
        update,
        wire_version=WIRE_VERSION,
        cursor=cursor,
    )
    assert snapshot is not None and snapshot["event"] == "snapshot"
    assert (
        event_from_update(
            "request",
            update,
            wire_version=WIRE_VERSION,
            cursor=cursor,
        )
        is None
    )
    terminal = event_from_update(
        "request",
        _update((), "", finished=True),
        wire_version=WIRE_VERSION,
        cursor=cursor,
    )
    assert terminal is not None
    assert terminal["sequence"] == 1
    assert terminal["finished"] is True


def test_v2_rejects_nonterminal_visible_text_rewrite_at_both_boundaries():
    cursor = WireEventCursor()
    accumulator = _WireAccumulator()
    snapshot = _encode(cursor, _update((1,), "abc"))
    accumulator.apply(snapshot)
    with pytest.raises(ValueError, match="already-emitted non-terminal prefix"):
        _encode(cursor, _update((1, 2), "MUTATED"))

    malformed = {
        "wire_version": WIRE_VERSION,
        "request_id": "request",
        "stream_id": "stream",
        "sequence": 1,
        "event": "delta",
        "output_offset": 1,
        "new_token_ids": [2],
        "text_offset": 0,
        "text_delta": "MUTATED",
        "num_cached_tokens": 4,
        "cumulative_logprob": -0.5,
        "finished": False,
        "finish_reason": None,
    }
    with pytest.raises(EngineServiceError, match="non-terminal visible-text rewrite"):
        accumulator.apply(malformed)


def test_v2_can_introduce_logprob_metadata_after_snapshot():
    cursor = WireEventCursor()
    accumulator = _WireAccumulator()
    snapshot = _encode(cursor, _update((1,), "x"))
    delta = _encode(
        cursor,
        _update(
            (1,),
            "x",
            finished=True,
            logprobs=(),
            content=(),
        ),
    )
    assert delta["new_logprobs"] == []
    assert delta["new_logprob_content"] == []
    accumulator.apply(snapshot)
    normalized = accumulator.apply(delta)
    assert normalized["logprobs"] == []
    assert normalized["logprob_content"] == []


@pytest.mark.parametrize(
    ("fields", "match"),
    (
        (("new_logprobs", "logprob_offset"), "omitted active logprob metadata"),
        (
            ("new_logprob_content", "logprob_content_offset"),
            "omitted active rich logprob metadata",
        ),
    ),
)
def test_v2_rejects_disappearing_active_logprob_metadata(fields, match):
    cursor = WireEventCursor()
    accumulator = _WireAccumulator()
    snapshot = _encode(
        cursor,
        _update(
            (1,),
            "x",
            logprobs=({1: -0.25},),
            content=(_content(1, "x"),),
        ),
    )
    terminal = _encode(
        cursor,
        _update(
            (1, 2),
            "xy",
            finished=True,
            logprobs=({1: -0.25}, {2: -0.25}),
            content=(_content(1, "x"), _content(2, "y")),
        ),
    )
    accumulator.apply(snapshot)
    for field in fields:
        terminal.pop(field)
    with pytest.raises(EngineServiceError, match=match):
        accumulator.apply(terminal)


def test_legacy_event_remains_cumulative_and_new_client_accepts_it():
    update = _update((1, 2), "legacy", finished=True)
    event = event_from_update(
        "request",
        update,
        wire_version=LEGACY_WIRE_VERSION,
    )
    assert event is not None
    assert "wire_version" not in event
    assert _WireAccumulator().apply(event) == event


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"sequence": 3}, "sequence mismatch"),
        ({"output_offset": 99}, "output_offset mismatch"),
        ({"text_offset": 99}, "text_offset mismatch"),
    ),
)
def test_v2_rejects_sequence_and_offset_gaps(mutation, match):
    cursor = WireEventCursor()
    accumulator = _WireAccumulator()
    accumulator.apply(_encode(cursor, _update((1,), "x")))
    delta = _encode(cursor, _update((1, 2), "xy"))
    delta.update(mutation)
    with pytest.raises(EngineServiceError, match=match):
        accumulator.apply(delta)


def test_v2_rejects_duplicate_snapshot_and_mid_request_version_change():
    cursor = WireEventCursor()
    snapshot = _encode(cursor, _update((1,), "x"))
    accumulator = _WireAccumulator()
    accumulator.apply(snapshot)
    with pytest.raises(EngineServiceError, match="duplicate snapshot"):
        duplicate = dict(snapshot)
        accumulator.apply(duplicate)
    legacy = event_from_update(
        "request",
        _update((1, 2), "xy"),
        wire_version=LEGACY_WIRE_VERSION,
    )
    assert legacy is not None
    with pytest.raises(EngineServiceError, match="changed version"):
        accumulator.apply(legacy)


def test_generate_style_reconstruction_materializes_only_the_terminal_text():
    cursor = WireEventCursor()
    accumulator = _WireAccumulator()
    total = 2_048
    for length in range(1, total + 1):
        event = _encode(
            cursor,
            _update(
                tuple(range(length)),
                "x" * length,
                finished=length == total,
            ),
        )
        normalized = accumulator.apply(
            event,
            materialize=event["finished"],
        )
        if not event["finished"]:
            assert normalized == {"finished": False}
    assert normalized["outputs"] == list(range(total))
    assert normalized["text"] == "x" * total
    assert normalized["finished"] is True
