"""Every Chat UI filter leaves an OpenAI Responses body (m11 D7).

Open WebUI v0.12.3 (as v0.11.0) forwards unknown top-level keys and replays
saved reasoning items verbatim when it converts its chat body for an
``api_type: responses`` connection, and Kairyu's /v1/responses rejects both
(replays 2026-10-17 and 2026-10-18: ``reasoning_effort`` and chat-era
``attributes`` answered 400). L4's filter serves every workspace, with or
without effort levels; the two orchestrated examples keep their own until
they move to L4.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
L4_FILTER = ROOT / "deploy/workspace/plugins/responses_filter.py"
LEVELS = ["default", "low", "high", "max"]
FILTERS = [
    pytest.param(L4_FILTER, LEVELS, id="l4-effort"),
    pytest.param(L4_FILTER, None, id="l4-plain"),
    *(
        pytest.param(path, None, id=path.parent.name)
        for path in sorted(ROOT.glob("examples/*/webui-*filter.py"))
    ),
]
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


def _load(path: Path, levels: list[str] | None, monkeypatch):
    # L4's filter reads the workspace's effort levels from the Chat UI's environment.
    monkeypatch.setenv("KAIRYU_WORKSPACE_EFFORT_LEVELS", json.dumps(levels or []))
    spec = importlib.util.spec_from_file_location(f"filter_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Filter()


@pytest.mark.parametrize("path,levels", FILTERS)
def test_filter_leaves_only_responses_fields(path: Path, levels, monkeypatch):
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

    result = _load(path, levels, monkeypatch).inlet(body, None)

    assert set(result) <= _CONVERTED | _RESPONSES_FIELDS, set(result) - _RESPONSES_FIELDS
    assert result["store"] is False
    assert result["reasoning"] == {"effort": "high"}
    replayed = result["messages"][1]["output"]
    assert [item["type"] for item in replayed] == ["reasoning", "message"]
    assert all(set(item) <= _ITEM_FIELDS[item["type"]] for item in replayed)


def test_effort_dropdown_default_restores_the_model_default(monkeypatch):
    """``default`` drops a saved ``reasoning_effort`` (deepseek-v4.1-flash-8gpu's
    default params carry high); a level becomes ``reasoning.effort``."""
    selector = _load(L4_FILTER, LEVELS, monkeypatch)

    def choose(effort):
        body = {"reasoning_effort": "high", "max_tokens": 32768}
        return selector.inlet(body, {"valves": selector.UserValves(reasoning_effort=effort)})

    assert choose("default") == {"max_tokens": 32768, "store": False}
    assert choose("low")["reasoning"] == {"effort": "low"}
