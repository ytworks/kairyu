"""Responses remote compaction: sealed summaries, storage, and failure modes."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import struct
import time

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi.testclient import TestClient

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.responses.compaction import CompactionCodec
from kairyu.entrypoints.server.responses.sealing import (
    SealingConfig,
    SealingKey,
    SealingKeyRing,
)
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
        assert token.startswith("kst2.")
        encoded = token.removeprefix("kst2.")
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
        tampered = "kst2." + base64.urlsafe_b64encode(tampered_payload).rstrip(
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
    for refused in (forged_response, tampered_response):
        _assert_refused_seal(refused, "not issued by this server")

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
    _assert_refused_seal(cross_tenant, "not issued by this server")


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


def _assert_refused_seal(response, reason: str) -> None:
    # Compaction fails closed (m20 D6): one typed 400 naming the item and
    # the remedy a Codex user needs, whatever the reason.
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_encrypted_content"
    assert error["param"] == "input[0].encrypted_content"
    assert reason in error["message"]
    assert "start a new session" in error["message"]


def _summary_backend(summary: str = "SEALED SUMMARY") -> MockBackend:
    return MockBackend({"compacted continuation": summary, "continue": "CONTINUED"})


def _gateway(stack, tmp_path, backend, *, primary=None, previous=None, **limits):
    settings = ServerSettings(
        responses_compaction_secret_env=primary,
        sealing=SealingConfig(previous_secrets_env=previous, **limits),
        usage_ledger_path=str(tmp_path / f"usage-{os.urandom(4).hex()}.jsonl"),
    )
    app = create_legacy_app({"m": backend}, settings=settings)
    return stack.enter_context(TestClient(app))


def _compaction(http):
    return http.post(
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


def _compact(http) -> str:
    response = _compaction(http)
    assert response.status_code == 200
    return response.json()["output"][0]["encrypted_content"]


def _continue(http, token: str):
    return http.post(
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


def test_sealing_key_rotation_is_two_phase(tmp_path, monkeypatch):
    # Promoting a new secret in one step strands every token a promoted
    # gateway issues on gateways not yet rolled (unknown key id). Phase 1
    # deploys it accept-only everywhere; phase 2 promotes it and keeps the
    # old secret accept-only until the old tokens are gone.
    monkeypatch.setenv("SEAL_OLD", "old-sealing-secret-0123456789abcdef")
    monkeypatch.setenv("SEAL_NEW", "new-sealing-secret-0123456789abcdef")
    phase_one_backend = _summary_backend()
    phase_two_backend = _summary_backend("NEW KEY SUMMARY")
    after_backend = _summary_backend()
    with contextlib.ExitStack() as stack:
        old_backend = _summary_backend("OLD KEY SUMMARY")
        before = _gateway(stack, tmp_path, old_backend, primary="SEAL_OLD")
        phase_one = _gateway(
            stack, tmp_path, phase_one_backend, primary="SEAL_OLD", previous="SEAL_NEW"
        )
        phase_two = _gateway(
            stack, tmp_path, phase_two_backend, primary="SEAL_NEW", previous="SEAL_OLD"
        )
        after = _gateway(stack, tmp_path, after_backend, primary="SEAL_NEW")
        old_token = _compact(before)
        new_token = _compact(phase_two)

        assert _continue(phase_one, new_token).status_code == 200
        assert "NEW KEY SUMMARY" in phase_one_backend.prompts_seen[-1]
        assert _continue(phase_two, old_token).status_code == 200
        assert "OLD KEY SUMMARY" in phase_two_backend.prompts_seen[-1]
        # A shared primary secret survives the gateway hop (and a restart).
        assert _continue(after, new_token).status_code == 200
        assert "NEW KEY SUMMARY" in after_backend.prompts_seen[-1]
        _assert_refused_seal(_continue(before, new_token), "no longer accepts")
        _assert_refused_seal(_continue(after, old_token), "no longer accepts")


def test_previous_release_kcp1_token_still_restores(tmp_path, monkeypatch):
    # Sessions compacted before the kst2 release replay their kcp1 token
    # after the upgrade; the same secret must still open it.
    secret = b"legacy-sealing-secret-0123456789abcdef"
    monkeypatch.setenv("SEAL_LEGACY", secret.decode())
    key = hashlib.sha256(b"kairyu.responses.compaction.key.v1\0" + secret).digest()
    nonce = os.urandom(12)
    sealed = AESGCM(key).encrypt(
        nonce, b"LEGACY SUMMARY", b"kairyu.responses.compaction.v1\0default"
    )
    token = "kcp1." + base64.urlsafe_b64encode(nonce + sealed).rstrip(b"=").decode()
    tampered = token[:-4] + ("AAAA" if token[-4:] != "AAAA" else "BBBB")
    backend = _summary_backend()
    with contextlib.ExitStack() as stack:
        http = _gateway(stack, tmp_path, backend, primary="SEAL_LEGACY")
        continued = _continue(http, token)
        refused = _continue(http, tampered)
    assert continued.status_code == 200
    assert "LEGACY SUMMARY" in backend.prompts_seen[-1]
    _assert_refused_seal(refused, "not issued by this server")


def test_sealed_item_size_cap_holds_when_issuing_and_opening(tmp_path):
    # The cap bounds the work a client can force: an oversized token is
    # refused for its size before base64 or AES-GCM ever runs on it. A summary
    # too large to reopen fails the compaction instead of handing the client
    # a token that strands its session on the next turn.
    backend = _summary_backend("x" * 65536)
    with contextlib.ExitStack() as stack:
        http = _gateway(stack, tmp_path, backend, sealed_item_max_bytes=65536)
        oversized = _continue(http, "kst2." + "!" * 65536)
        unsealable = _compaction(http)
    _assert_refused_seal(oversized, "exceeds the 65536-byte")
    assert unsealable.status_code == 502
    assert unsealable.json()["error"]["code"] == "compaction_failed"


def test_sealed_item_max_age_refuses_expired_tokens(tmp_path, monkeypatch):
    secret = "aging-sealing-secret-0123456789abcdef"
    monkeypatch.setenv("SEAL_AGING", secret)
    hour_ago = SealingKeyRing(
        SealingKey.from_secret(secret.encode()), clock=lambda: time.time() - 3600
    )
    stale_token = CompactionCodec(hour_ago).encode("STALE SUMMARY", owner="default")
    raw = bytearray(_unpadded_b64decode(stale_token.removeprefix("kst2.")))
    raw[10:18] = struct.pack(">Q", int(time.time()))  # header issued_at
    restamped = "kst2." + base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    backend = _summary_backend("FRESH SUMMARY")
    with contextlib.ExitStack() as stack:
        http = _gateway(
            stack, tmp_path, backend, primary="SEAL_AGING", sealed_max_age_s=600
        )
        fresh = _continue(http, _compact(http))
        stale = _continue(http, stale_token)
        forged_fresh = _continue(http, restamped)
    assert fresh.status_code == 200
    assert "FRESH SUMMARY" in backend.prompts_seen[-1]
    _assert_refused_seal(stale, "expired")
    _assert_refused_seal(forged_fresh, "not issued by this server")


def _unpadded_b64decode(encoded: str) -> bytes:
    return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
