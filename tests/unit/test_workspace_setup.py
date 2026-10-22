"""L4 workspace setup against a running Open WebUI (deploy/workspace).

Setup installs the one Responses filter and the profile's tools and refuses a
Chat UI whose configuration differs from the workspace. The CPU demo's CI run
sets up a fresh Open WebUI, so these cases cover what it cannot reach: an
Open WebUI that already holds the examples' earlier filters, and drift.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "deploy" / "workspace"))
import kairyu_workspace  # noqa: E402
from kairyu_workspace.spec import DEFAULT_CAPABILITIES  # noqa: E402

SECTION = {
    "profile": "ja-v1",
    "port": 3018,
    "bind": "127.0.0.1",
    "auth": "none",
    "models": ["m"],
    "default_params": {"max_tokens": 1024},
    "reasoning_effort": {"levels": ["default", "low", "high"], "help": "effort"},
    "cpus": {"chat_ui": 8, "docling": 4, "docling_workers": 2},
}


class FakeOpenWebUI(ThreadingHTTPServer):
    """The admin API routes setup uses, with an Open WebUI configured from env."""

    def __init__(self, workspace: kairyu_workspace.Workspace) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        retrieval = workspace.profile["retrieval"]
        interpreter = workspace.profile["code_interpreter"]
        self.levels = list(workspace.effort_levels)
        self.functions: dict[str, dict] = {}
        self.tools: dict[str, dict] = {}
        self.calls: list[str] = []
        self.retrieval = {
            "CONTENT_EXTRACTION_ENGINE": "docling",
            "DOCLING_SERVER_URL": "http://docling:5001",
            "DOCLING_PARAMS": workspace.profile["docling"]["params"],
            "RAG_RERANKING_ENGINE": "",
            "RAG_RERANKING_MODEL": "/opt/kairyu-workspace-models/bge-reranker-v2-m3",
            "ENABLE_RAG_HYBRID_SEARCH": True,
            "HYBRID_BM25_WEIGHT": retrieval["hybrid_bm25_weight"],
            "TOP_K": retrieval["top_k"],
            "TOP_K_RERANKER": retrieval["top_k_reranker"],
            "CHUNK_SIZE": retrieval["chunk_size"],
            "CHUNK_OVERLAP": retrieval["chunk_overlap"],
        }
        self.meta = {
            "capabilities": DEFAULT_CAPABILITIES,
            "defaultFeatureIds": interpreter["default_features"],
            "toolIds": interpreter["default_tools"],
        }

    def answer(self, method: str, path: str, body: dict | None):
        self.calls.append(f"{method} {path}")
        functions = "/api/v1/functions/id/"
        if path == "/api/v1/auths/signin":
            return {"token": "t", "role": "admin"}
        if path == "/api/version":
            return {"version": "0.12.3"}
        if path == "/openai/config":
            return {
                "OPENAI_API_BASE_URLS": ["http://kairyu:8000/v1"],
                "OPENAI_API_CONFIGS": {"0": {"api_type": "responses"}},
            }
        if path in ("/api/v1/functions/", "/api/v1/functions/export"):
            return list(self.functions.values())
        if path == "/api/v1/functions/create":
            self.functions[body["id"]] = {**body, "is_active": False, "is_global": False}
            return self.functions[body["id"]]
        if path.startswith(functions):
            function_id, action = path[len(functions) :].split("/", 1)
            row = self.functions[function_id]
            if action == "delete":
                del self.functions[function_id]
                return True
            if action == "update":
                row.update(body)
            elif action == "toggle":
                row["is_active"] = not row["is_active"]
            elif action == "toggle/global":
                row["is_global"] = not row["is_global"]
            elif action == "valves/user/spec":
                return {"properties": {"reasoning_effort": {"enum": self.levels}}}
            return row
        if path in ("/api/v1/tools/", "/api/v1/tools/export"):
            return list(self.tools.values())
        if path.endswith("/access/update"):
            tool_id = path.split("/")[-3]
            self.tools[tool_id]["access_grants"] = body["access_grants"]
            return self.tools[tool_id]
        if path == "/api/v1/tools/create" or path.endswith("/update"):
            self.tools[body["id"]] = {**body, "specs": [{"name": "provide_download"}]}
            return self.tools[body["id"]]
        if path == "/api/models":
            return {"data": [{"id": "m", "info": {"meta": self.meta}}]}
        if path == "/api/v1/retrieval/config":
            return self.retrieval
        if path == "/api/v1/retrieval/embedding":
            return {
                "RAG_EMBEDDING_ENGINE": "",
                "RAG_EMBEDDING_MODEL": "/opt/kairyu-workspace-models/bge-m3",
            }
        if path == "/api/config":
            return {
                "features": {
                    "enable_code_interpreter": True,
                    "enable_pyodide_file_persistence": False,
                },
                "code": {"interpreter_engine": "pyodide"},
            }
        raise AssertionError(f"unexpected route {method} {path}")


class _Handler(BaseHTTPRequestHandler):
    def _reply(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        payload = json.dumps(self.server.answer(self.command, self.path, body)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_DELETE = _reply

    def log_message(self, *_: object) -> None:
        pass


@pytest.fixture
def open_webui():
    server = FakeOpenWebUI(kairyu_workspace.load(SECTION))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def _url(server: FakeOpenWebUI) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_setup_replaces_earlier_filters_and_keeps_an_active_one_on(open_webui):
    for function_id in ("reasoning_effort", "kairyu_responses"):
        open_webui.functions[function_id] = {
            "id": function_id,
            "content": "old",
            "is_active": True,
            "is_global": True,
        }

    kairyu_workspace.setup(_url(open_webui), kairyu_workspace.load(SECTION))

    assert set(open_webui.functions) == {"kairyu_responses"}
    assert open_webui.functions["kairyu_responses"]["is_active"] is True
    assert open_webui.functions["kairyu_responses"]["is_global"] is True
    assert not any(call.endswith(("/toggle", "/toggle/global")) for call in open_webui.calls)
    assert open_webui.functions["kairyu_responses"]["content"].startswith('"""\ntitle:')
    assert set(open_webui.tools) == {"file_download"}
    # Other signed-in users get the download tool too (auth-on workspaces).
    assert open_webui.tools["file_download"]["access_grants"] == [
        {"principal_type": "user", "principal_id": "*", "permission": "read"}
    ]


def test_setup_fails_closed_naming_the_drifted_setting(open_webui):
    open_webui.retrieval["CHUNK_SIZE"] = 1000

    with pytest.raises(SystemExit, match="CHUNK_SIZE"):
        kairyu_workspace.setup(_url(open_webui), kairyu_workspace.load(SECTION))
