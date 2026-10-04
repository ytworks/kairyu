"""Responses compatibility with the official OpenAI SDK (strict validation)."""

from __future__ import annotations

import pytest

from tests.server.live_server import async_openai_client, openai_client
from tests.server.responses._helpers import LengthBackend, _app


def test_official_sdk_typed_text_stream_and_final_response(tmp_path):
    with openai_client(_app(tmp_path)) as client:
        with client.responses.stream(model="m", input="hello") as stream:
            events = list(stream)
            final = stream.get_final_response()

    types = [event.type for event in events]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert final.status == "completed"
    assert final.output_text == "streamed hello"
    assert final.usage.input_tokens_details.cached_tokens == 0
    assert final.usage.output_tokens_details.reasoning_tokens == 0


def test_create_stream_true_returns_sdk_typed_events(tmp_path):
    with openai_client(_app(tmp_path)) as client:
        events = list(client.responses.create(model="m", input="hello", stream=True))
    assert events[0].type == "response.created"
    assert events[-1].type == "response.completed"
    deltas = [
        event.delta for event in events if event.type == "response.output_text.delta"
    ]
    assert "".join(deltas) == "streamed hello"


@pytest.mark.asyncio
async def test_official_async_sdk_stream_is_typed(tmp_path):
    async with async_openai_client(_app(tmp_path)) as client:
        stream = await client.responses.create(model="m", input="hello", stream=True)
        events = [event async for event in stream]
    assert events[0].type == "response.created"
    assert events[-1].type == "response.completed"
    assert events[-1].response.output_text == "streamed hello"


def test_incomplete_stream_has_consistent_item_and_terminal_status(tmp_path):
    backend = LengthBackend({"hello": "truncated"})
    with openai_client(_app(tmp_path, backend)) as client:
        events = list(client.responses.create(model="m", input="hello", stream=True))

    item_done = next(event for event in events if event.type == "response.output_item.done")
    terminal = events[-1]
    assert item_done.item.status == "incomplete"
    assert terminal.type == "response.incomplete"
    assert terminal.response.status == "incomplete"
    assert terminal.response.output[0].status == "incomplete"
    assert terminal.response.incomplete_details.reason == "max_output_tokens"


def test_incomplete_unary_has_consistent_item_and_response_status(tmp_path):
    backend = LengthBackend({"hello": "truncated"})
    with openai_client(_app(tmp_path, backend)) as client:
        response = client.responses.create(model="m", input="hello")

    assert response.status == "incomplete"
    assert response.output[0].status == "incomplete"
    assert response.incomplete_details.reason == "max_output_tokens"
