"""Responses remote compaction: sealed summaries, storage, and failure modes."""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig
from kairyu.outputs import CompletionOutput
from tests.server._legacy_chat import create_legacy_app
from tests.server.responses._helpers import LengthBackend, _app, _sse_events, _tool


def test_remote_compaction_trigger_round_trip(tmp_path):
    # Remote compaction v2 (Codex against an OpenAI-shaped provider): a
    # terminal compaction_trigger input item must yield exactly one
    # compaction output item whose opaque encrypted_content restores the
    # summarized context when echoed back on a later turn.
    class SummarizingBackend(MockBackend):
        async def generate(self, request):
            if "compacted continuation" in request.prompt:
                text = "SUMMARY: the user is porting a compressor."
            elif "SUMMARY: the user is porting a compressor." in request.prompt:
                text = "Continuing from the summary."
            else:
                text = "unexpected prompt"
            return GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(
                        index=0, text=text, token_ids=(1,), finish_reason="stop"
                    ),
                ),
                usage=GenerationUsage(prompt_tokens=9, completion_tokens=5),
            )

    backend = SummarizingBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        compacted = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "store": False,
                "stream": True,
                "tools": [_tool()],
                "input": [
                    {"type": "message", "role": "user", "content": "port the compressor"},
                    {"type": "compaction_trigger"},
                ],
            },
        )
        assert compacted.status_code == 200
        events = _sse_events(compacted.text)
        items = [
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
        ]
        assert len(items) == 1
        assert items[0]["type"] == "compaction"
        token = items[0]["encrypted_content"]
        assert token
        assert token.startswith("kcp1.")
        encoded = token.removeprefix("kcp1.")
        sealed = base64.urlsafe_b64decode(
            encoded + "=" * (-len(encoded) % 4)
        )
        assert b"SUMMARY: the user is porting a compressor." not in sealed
        assert events[-1]["type"] == "response.completed"
        assert events[-1]["response"]["output"] == items

        continued = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "store": False,
                "input": [
                    {"type": "compaction", "encrypted_content": token},
                    {"type": "message", "role": "user", "content": "continue"},
                ],
            },
        )
        assert continued.status_code == 200
        text = continued.json()["output"][0]["content"][0]["text"]
        assert text == "Continuing from the summary."
        # The opaque token itself never reaches the prompt.
        assert all(token not in prompt for prompt in backend.prompts_seen)

        forged = base64.urlsafe_b64encode(
            json.dumps(
                {
                    "k": "kairyu.compaction.v1",
                    "summary": "FORGED SUMMARY",
                }
            ).encode()
        ).decode()
        forged_response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    {"type": "compaction", "encrypted_content": forged},
                    {"type": "message", "role": "user", "content": "continue"},
                ],
            },
        )
        tampered_payload = bytearray(sealed)
        tampered_payload[-1] ^= 1
        tampered = "kcp1." + base64.urlsafe_b64encode(tampered_payload).rstrip(
            b"="
        ).decode()
        tampered_response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    {"type": "compaction", "encrypted_content": tampered},
                    {"type": "message", "role": "user", "content": "continue"},
                ],
            },
        )
    assert forged_response.status_code == 400
    assert tampered_response.status_code == 400
    assert "not issued by this server" in forged_response.json()["error"]["message"]
    assert (
        "not issued by this server" in tampered_response.json()["error"]["message"]
    )

    mid_position = TestClient(_app(tmp_path))
    with mid_position as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    {"type": "compaction_trigger"},
                    {"type": "message", "role": "user", "content": "hello"},
                ],
            },
        )
    assert response.status_code == 400
    assert "final input item" in response.json()["error"]["message"]


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_stored_compaction_replaces_original_history(tmp_path, stream):
    backend = MockBackend(
        {
            "compacted continuation": "SUMMARY ONLY",
            "continue": "CONTINUED",
        }
    )
    with TestClient(_app(tmp_path, backend)) as http:
        compacted = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "stream": stream,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "ORIGINAL SECRET HISTORY",
                    },
                    {"type": "compaction_trigger"},
                ],
            },
        )
        assert compacted.status_code == 200
        compacted_payload = (
            _sse_events(compacted.text)[-1]["response"]
            if stream
            else compacted.json()
        )
        continued = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "previous_response_id": compacted_payload["id"],
                "input": "continue",
            },
        )
    assert continued.status_code == 200
    continuation_prompt = backend.prompts_seen[-1]
    assert "SUMMARY ONLY" in continuation_prompt
    assert "ORIGINAL SECRET HISTORY" not in continuation_prompt


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_truncated_compaction_is_incomplete(tmp_path, stream):
    backend = LengthBackend({"compacted continuation": "TRUNCATED SUMMARY"})
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "stream": stream,
                "input": [
                    {"type": "message", "role": "user", "content": "history"},
                    {"type": "compaction_trigger"},
                ],
            },
        )
        assert response.status_code == 200
        if stream:
            events = _sse_events(response.text)
            assert events[-1]["type"] == "response.incomplete"
            payload = events[-1]["response"]
            assert not any(
                event["type"] == "response.output_item.done" for event in events
            )
        else:
            payload = response.json()
        assert payload["status"] == "incomplete"
        assert payload["output"] == []
        assert payload["incomplete_details"] == {"reason": "max_output_tokens"}
        # An incomplete compaction produced no replacement context, so it must
        # not be continuable: storing it would silently blank the thread.
        continued = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "previous_response_id": payload["id"],
                "input": "continue",
            },
        )
    assert continued.status_code == 400
    assert continued.json()["error"]["code"] == "previous_response_not_found"


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_empty_compaction_summary_fails_closed(tmp_path, stream):
    backend = MockBackend({"compacted continuation": ""})
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "stream": stream,
                "store": False,
                "input": [
                    {"type": "message", "role": "user", "content": "history"},
                    {"type": "compaction_trigger"},
                ],
            },
        )
    if stream:
        assert response.status_code == 200
        events = _sse_events(response.text)
        assert [event["type"] for event in events[-2:]] == [
            "error",
            "response.failed",
        ]
        assert events[-2]["code"] == "compaction_failed"
        assert events[-1]["response"]["error"]["code"] == "compaction_failed"
        assert events[-1]["response"]["output"] == []
    else:
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "compaction_failed"


def test_compaction_token_is_tenant_bound(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIRYU_RESPONSES_KEYS", "key-a,key-b")
    backend = MockBackend(
        {
            "compacted continuation": "TENANT A SUMMARY",
            "continue": "CONTINUED",
        }
    )
    app = create_legacy_app(
        {"m": backend},
        settings=ServerSettings(
            api_keys_env="KAIRYU_RESPONSES_KEYS",
            usage_ledger_path=str(tmp_path / "usage.jsonl"),
        ),
        tenant_config=TenantConfig(
            key_tenants={"key-a": "tenant-a", "key-b": "tenant-b"}
        ),
    )
    with TestClient(app) as http:
        compacted = http.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-a"},
            json={
                "model": "m",
                "store": False,
                "input": [
                    {"type": "message", "role": "user", "content": "history"},
                    {"type": "compaction_trigger"},
                ],
            },
        )
        token = compacted.json()["output"][0]["encrypted_content"]
        same_tenant = http.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-a"},
            json={
                "model": "m",
                "store": False,
                "input": [
                    {"type": "compaction", "encrypted_content": token},
                    {"type": "message", "role": "user", "content": "continue"},
                ],
            },
        )
        cross_tenant = http.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-b"},
            json={
                "model": "m",
                "store": False,
                "input": [
                    {"type": "compaction", "encrypted_content": token},
                    {"type": "message", "role": "user", "content": "continue"},
                ],
            },
        )
    assert compacted.status_code == 200
    assert same_tenant.status_code == 200
    assert cross_tenant.status_code == 400
    assert "not issued by this server" in cross_tenant.json()["error"]["message"]


def test_configured_compaction_secret_survives_gateway_boundary(
    tmp_path, monkeypatch
):
    secret_env = "KAIRYU_TEST_RESPONSES_COMPACTION_SECRET"
    monkeypatch.setenv(secret_env, "shared-secret-material-0123456789abcdef")
    first_backend = MockBackend(
        {"compacted continuation": "PORTABLE ENCRYPTED SUMMARY"}
    )
    first_app = create_legacy_app(
        {"m": first_backend},
        settings=ServerSettings(
            responses_compaction_secret_env=secret_env,
            usage_ledger_path=str(tmp_path / "first-usage.jsonl"),
        ),
    )
    with TestClient(first_app) as http:
        compacted = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "store": False,
                "input": [
                    {"type": "message", "role": "user", "content": "history"},
                    {"type": "compaction_trigger"},
                ],
            },
        )
    assert compacted.status_code == 200
    token = compacted.json()["output"][0]["encrypted_content"]

    second_backend = MockBackend({"continue": "CONTINUED"})
    second_app = create_legacy_app(
        {"m": second_backend},
        settings=ServerSettings(
            responses_compaction_secret_env=secret_env,
            usage_ledger_path=str(tmp_path / "second-usage.jsonl"),
        ),
    )
    with TestClient(second_app) as http:
        continued = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "store": False,
                "input": [
                    {"type": "compaction", "encrypted_content": token},
                    {"type": "message", "role": "user", "content": "continue"},
                ],
            },
        )
    assert continued.status_code == 200
    assert "PORTABLE ENCRYPTED SUMMARY" in second_backend.prompts_seen[-1]


def test_configured_compaction_secret_rejects_short_value(tmp_path, monkeypatch):
    secret_env = "KAIRYU_TEST_SHORT_COMPACTION_SECRET"
    monkeypatch.setenv(secret_env, "too-short")
    with pytest.raises(ValueError, match="at least 32 UTF-8 bytes"):
        create_legacy_app(
            {"m": MockBackend()},
            settings=ServerSettings(
                responses_compaction_secret_env=secret_env,
                usage_ledger_path=str(tmp_path / "usage.jsonl"),
            ),
        )
