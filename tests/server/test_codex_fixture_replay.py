"""Replay Codex wire fixtures against the server (M20 WP-02, D1).

Each file under ``tests/fixtures/codex/rust-v<ver>/`` (``extensions.json``
aside) is one Codex wire contract: a request Codex sends -- recorded through
``scripts/codex_gate/record_proxy.py`` or derived from codex-rs at the tag, see
its ``provenance`` -- the model behavior scripted for ``ScenarioBackend``, and
the expected outcome. The schema gate (``tests/contracts``) validates every
exchange; this test asserts the status, terminal event, output items and what
reached the backend. A fixture whose behavior a later work package delivers
carries ``replay.xfail`` and runs as ``xfail(strict=True)``, so that work
package has to flip it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.router import RouteThresholds, RuleRouter
from tests.server._legacy_chat import create_legacy_app
from tests.support.scenario_backend import ScenarioBackend, engine_pieces
from tests.support.scenario_script import (
    Part,
    Scenario,
    ToolCall,
    Turn,
    chunk_pieces,
    numbered_text,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "codex"
INVENTORY = "extensions.json"
TERMINAL_EVENTS = frozenset({"response.completed", "response.incomplete", "response.failed"})
# Fixtures script one generation per request, so AUTO always takes a direct tier.
_DIRECT_ROUTE = RouteThresholds(
    multi_step_markers=10**6,
    multi_agent_min_chars=10**9,
    reasoning_keywords=10**6,
    math_symbols=10**6,
    tier2_min_chars=10**9,
)
REPLAY_KEYS = frozenset({"model", "turns", "content_encoding", "expect", "xfail"})
EXPECT_KEYS = frozenset(
    {
        "status",
        "error",
        "terminal",
        "response_status",
        "output_types",
        "output_contains",
        "output_items",
        "min_output_tokens",
        "response_omits",
        "prompt_contains",
        "backend_tools_include",
        "backend_tools_exclude",
        "backend_tools_none",
        "backend_reasoning_effort",
    }
)
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
# Single segment, 8-byte content size, no checksum, no dictionary.
_ZSTD_FRAME_HEADER = 0xE0
_ZSTD_MAX_BLOCK = 128 * 1024


def _checked(path: Path) -> dict:
    """Load a fixture, failing collection on a key the replay does not read."""

    fixture = json.loads(path.read_text(encoding="utf-8"))
    replay = fixture["replay"]
    unknown = (set(replay) - REPLAY_KEYS) | (set(replay["expect"]) - EXPECT_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown replay keys {sorted(unknown)}")
    if fixture["request"]["body"].get("stream") and replay["expect"]["status"] < 400:
        if "terminal" not in replay["expect"]:
            raise ValueError(f"{path}: a streamed fixture names its terminal event")
    return fixture


def _fixture_params() -> Iterator[Any]:
    for path in sorted(FIXTURE_ROOT.glob("rust-v*/*.json")):
        if path.name == INVENTORY:
            continue
        fixture = _checked(path)
        xfail = fixture["replay"].get("xfail")
        marks = (
            [
                pytest.mark.xfail(
                    strict=True, reason=f"{xfail['gap']} / {xfail['wp']}: {xfail['reason']}"
                )
            ]
            if xfail
            else []
        )
        yield pytest.param(fixture, id=f"{path.parent.name}/{path.stem}", marks=marks)


def _part(spec: Mapping[str, Any]) -> Part:
    if "text" in spec:
        return spec["text"]
    if "tokens" in spec:
        return numbered_text(spec["tokens"], word=spec.get("word", "w"))
    if "tool_call" in spec:
        call = spec["tool_call"]
        return ToolCall(call["name"], json.dumps(call.get("arguments", {})))
    raise ValueError(f"unknown turn part {sorted(spec)}")


def _turn(spec: Mapping[str, Any]) -> Turn:
    turn = Turn(
        parts=tuple(_part(part) for part in spec["parts"]),
        finish_reason=spec.get("finish_reason", "stop"),
        chunk_tokens=spec.get("chunk_tokens", 1),
    )
    fail_after_parts = spec.get("fail_after_parts")
    if fail_after_parts is None:
        return turn
    # Fail once every chunk of the first N parts has been streamed.
    chunks = chunk_pieces(engine_pieces(turn), turn.chunk_tokens)
    complete = sum(1 for chunk in chunks if chunk[0].part < fail_after_parts)
    return Turn(
        turn.parts, turn.finish_reason, chunk_tokens=turn.chunk_tokens, fail_after_chunks=complete
    )


def _app(tmp_path: Path, replay: Mapping[str, Any], model: str, backend: ScenarioBackend):
    settings = ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl"))
    if replay["model"] == "auto":
        orchestrator = Orchestrator(
            {"tier1": backend, "tier2": backend}, router=RuleRouter(_DIRECT_ROUTE)
        )
        return create_legacy_app({}, orchestrators={model: orchestrator}, settings=settings)
    return create_legacy_app({model: backend}, settings=settings)


def _zstd_raw_frame(data: bytes) -> bytes:
    """A valid zstd frame of stored (raw) blocks; needs no zstd library."""

    step = _ZSTD_MAX_BLOCK
    blocks = [data[i : i + step] for i in range(0, len(data), step)] or [b""]
    frame = bytearray(_ZSTD_MAGIC + bytes([_ZSTD_FRAME_HEADER]) + len(data).to_bytes(8, "little"))
    for index, block in enumerate(blocks):
        last = index == len(blocks) - 1  # block header: last flag, type 0 (raw), size
        frame += ((len(block) << 3) | int(last)).to_bytes(3, "little") + block
    return bytes(frame)


def _final_response(response, *, stream: bool, expect: Mapping[str, Any]) -> dict:
    if not stream:
        return response.json()
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    terminal = [event for event in events if event["type"] in TERMINAL_EVENTS]
    assert len(terminal) == 1, [event["type"] for event in events]
    assert terminal[0]["type"] == expect["terminal"]
    return terminal[0]["response"]


def _assert_backend(backend: ScenarioBackend, expect: Mapping[str, Any]) -> None:
    calls = backend.calls
    prompts = "\n".join(call.prompt for call in calls)
    for text in expect.get("prompt_contains", ()):
        assert text in prompts, text
    offered = {tool["function"]["name"] for call in calls for tool in call.request.tools}
    assert set(expect.get("backend_tools_include", ())) <= offered, sorted(offered)
    assert not set(expect.get("backend_tools_exclude", ())) & offered, sorted(offered)
    if expect.get("backend_tools_none"):
        assert calls and not offered, sorted(offered)
    if "backend_reasoning_effort" in expect:
        efforts = {call.request.reasoning_effort for call in calls}
        assert efforts == {expect["backend_reasoning_effort"]}, efforts


@pytest.mark.parametrize("fixture", list(_fixture_params()))
def test_codex_fixture_replay(fixture, tmp_path):
    request, replay = fixture["request"], fixture["replay"]
    expect = replay["expect"]
    body = request["body"]
    backend = ScenarioBackend(Scenario(turns=tuple(_turn(turn) for turn in replay["turns"])))
    content = json.dumps(body).encode()
    headers = dict(request["headers"])
    if replay.get("content_encoding") == "zstd":
        content = _zstd_raw_frame(content)
        headers["content-encoding"] = "zstd"

    with TestClient(_app(tmp_path, replay, body["model"], backend)) as client:
        response = client.request(
            request["method"], request["path"], content=content, headers=headers
        )

    assert response.status_code == expect["status"], response.text[:500]
    if expect["status"] >= 400:
        error = response.json()["error"]
        assert {key: error.get(key) for key in expect["error"]} == expect["error"]
        return
    final = _final_response(response, stream=bool(body.get("stream")), expect=expect)
    if "response_status" in expect:
        assert final["status"] == expect["response_status"]
    output = final.get("output", [])
    if "output_types" in expect:
        assert [item["type"] for item in output] == expect["output_types"]
    for kind in expect.get("output_contains", ()):
        assert kind in [item["type"] for item in output], output
    if "output_items" in expect:
        assert [
            {key: item.get(key) for key in wanted}
            for item, wanted in zip(output, expect["output_items"], strict=True)
        ] == expect["output_items"]
    for item in output:
        if item["type"] == "function_call":  # Codex pairs the tool output by call_id.
            assert item["call_id"], item
    usage = final.get("usage")
    if usage is not None:  # Codex sizes its context window from these counts.
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    if "min_output_tokens" in expect:
        assert usage["output_tokens"] >= expect["min_output_tokens"]
    for key in expect.get("response_omits", ()):
        assert key not in final
    _assert_backend(backend, expect)
