"""Every Chat UI filter (examples, compose demo) leaves an OpenAI Responses body (m11 D7).

Open WebUI v0.12.3 (as v0.11.0) forwards unknown top-level keys and replays
saved reasoning items verbatim when it converts its chat body for an
``api_type: responses`` connection, and Kairyu's /v1/responses rejects both
(replays 2026-10-17 and 2026-10-18: ``reasoning_effort`` and chat-era
``attributes`` answered 400).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILTERS = sorted(
    [*ROOT.glob("examples/*/webui-*filter.py"), ROOT / "deploy/compose/webui-responses-filter.py"]
)
# Keys the pinned Open WebUI either converts (messages, max_tokens) or drops
# itself (stream_options, presence_penalty, ...); everything else must already
# be a Responses field when the filter returns.
_CONVERTED = {"messages", "max_tokens", "stream_options", "presence_penalty"}
_RESPONSES_FIELDS = {"model", "stream", "temperature", "top_p", "reasoning", "store", "tools"}
_ITEM_FIELDS = {
    "message": {"type", "role", "content"},
    "reasoning": {"type", "id", "summary", "content", "encrypted_content", "status"},
    "function_call": {"type", "id", "call_id", "name", "arguments"},
}


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"filter_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Filter()


def test_every_non_jev_example_with_a_chat_ui_ships_a_filter():
    assert {path.parent.name for path in FILTERS} >= {
        "deepseek-v4-flash-0731-8gpu",
        "deepseek-v4-flash-0731-dp2-8gpu",
        "qwen3.8-27b-1gpu",
        "qwen3.8-27b-dp8-8gpu",
    }


@pytest.mark.parametrize("path", FILTERS, ids=lambda path: path.parent.name)
def test_filter_leaves_only_responses_fields(path: Path):
    body = {
        "model": "m",
        "stream": True,
        "max_tokens": 4096,
        "temperature": 1.0,
        "reasoning_effort": "high",
        "seed": 7,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "hello",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": "t"}],
                        "attributes": {"duration": 1},
                    },
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                        "started_at": 1,
                    },
                    {"type": "open_webui_status", "text": "done"},
                ],
            },
            {"role": "user", "content": "again"},
        ],
    }

    result = _load(path).inlet(body, None)

    assert set(result) <= _CONVERTED | _RESPONSES_FIELDS, set(result) - _RESPONSES_FIELDS
    assert result["store"] is False
    assert result["reasoning"] == {"effort": "high"}
    replayed = result["messages"][1]["output"]
    assert [item["type"] for item in replayed] == ["reasoning", "message"]
    assert all(set(item) <= _ITEM_FIELDS[item["type"]] for item in replayed)
