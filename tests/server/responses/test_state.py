"""Responses stored state: what previous_response_id can and cannot read back."""

from __future__ import annotations

from fastapi.testclient import TestClient

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig
from tests.server._legacy_chat import create_legacy_app
from tests.server.responses._helpers import _app, _sse_events, _tool


def test_phase_and_opaque_function_arguments_survive_stateless_history(tmp_path):
    backend = MockBackend({"tool output": "done"})
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": "working",
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_opaque",
                        "name": "add",
                        "arguments": "provider-specific opaque arguments",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_opaque",
                        "output": "tool output",
                    },
                ],
                "tools": [_tool()],
            },
        )
        stored_items = http.app.state.response_store.get(response.json()["id"])
    assert response.status_code == 200
    assert "provider-specific opaque arguments" in backend.prompts_seen[0]
    assert stored_items is not None
    assert stored_items[0]["phase"] == "commentary"


def test_store_false_and_cross_tenant_stream_ids_are_not_readable(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KAIRYU_RESPONSES_KEYS", "key-a,key-b")
    app = create_legacy_app(
        {"m": MockBackend({"hello": "tenant response"})},
        settings=ServerSettings(
            api_keys_env="KAIRYU_RESPONSES_KEYS",
            usage_ledger_path=str(tmp_path / "usage.jsonl"),
        ),
        tenant_config=TenantConfig(
            key_tenants={"key-a": "tenant-a", "key-b": "tenant-b"}
        ),
    )
    with TestClient(app) as http:
        first = http.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-a"},
            json={"model": "m", "input": "hello", "stream": True},
        )
        response_id = _sse_events(first.text)[-1]["response"]["id"]
        cross_tenant = http.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-b"},
            json={
                "model": "m",
                "input": "again",
                "previous_response_id": response_id,
            },
        )
        unstored = http.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-a"},
            json={"model": "m", "input": "hello", "store": False},
        )
        not_found = http.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-a"},
            json={
                "model": "m",
                "input": "again",
                "previous_response_id": unstored.json()["id"],
            },
        )
    assert first.status_code == 200
    # Tenant-blind (D-d): another tenant's id answers like an unknown one.
    assert cross_tenant.status_code == not_found.status_code == 400
    assert cross_tenant.json()["error"]["code"] == not_found.json()["error"]["code"]
