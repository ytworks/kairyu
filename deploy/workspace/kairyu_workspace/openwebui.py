"""Open WebUI setup for an L4 workspace: plugins installed, configuration checked.

Setup signs in as the admin, checks the Open WebUI version, keeps Kairyu's
connection on the Responses API, installs the Responses filter and the
profile's default tools, and refuses to finish while any retrieval,
code-interpreter or model setting differs from the workspace (fail closed:
an example's ``run.sh up`` stops with the differing keys).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .spec import (
    DEFAULT_CAPABILITIES,
    DOCLING_URL,
    OPEN_WEBUI_VERSION,
    PLUGINS,
    TOOLS,
    Workspace,
    model_path,
)

FILTER_ID = "kairyu_responses"
# The filters examples and the CPU demo installed before L4 (each example
# had its own copy); setup removes them so exactly one filter is global.
RETIRED_FILTER_IDS = ("reasoning_effort", "responses_api")
KAIRYU_CONNECTION_URL = "http://kairyu:8000/v1"
_EMBEDDING_KEYS = ("RAG_EMBEDDING_ENGINE", "RAG_EMBEDDING_MODEL")
# Every signed-in user may use a tool L4 installs: Open WebUI keeps an admin's
# tool without grants private and leaves it out of other users' chats.
USERS_READ = {"principal_type": "user", "principal_id": "*", "permission": "read"}
_TOOL_META = {
    "file_download": ("File download", "Give the user a download link for a file the code wrote.")
}


def api(ui_url: str, path: str, *, token: str | None = None, payload=None, method=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{ui_url}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method=method or ("POST" if payload is not None else "GET"),
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read() or b"null")


def sign_in(ui_url: str, auth: str, email: str = "", password: str = "") -> str:
    """An admin session token: the single no-auth user, or the named admin."""
    if auth == "admin" and not (email and password):
        raise SystemExit("WEBUI_ADMIN_EMAIL and WEBUI_ADMIN_PASSWORD are required to set up")
    credentials = {"email": email, "password": password} if auth == "admin" else {}
    signin = api(
        ui_url, "/api/v1/auths/signin", payload={"email": "", "password": "", **credentials}
    )
    if not isinstance(signin, dict) or not signin.get("token"):
        raise SystemExit("Open WebUI sign-in did not return a session token")
    if signin.get("role") != "admin":
        raise SystemExit("Open WebUI setup must sign in as the admin user")
    return signin["token"]


def _differences(observed: dict, expected: dict) -> dict:
    return {
        key: (observed[key], expected[key]) for key in expected if observed[key] != expected[key]
    }


def _ensure_responses_connection(ui_url: str, token: str) -> None:
    """Kairyu's connection uses the Responses API (m11 D7 amendment 2026-10-17)."""
    config = api(ui_url, "/openai/config", token=token)
    urls = list(config.get("OPENAI_API_BASE_URLS") or [])
    if KAIRYU_CONNECTION_URL not in urls:
        raise SystemExit(f"Open WebUI has no connection to {KAIRYU_CONNECTION_URL}: {urls!r}")
    index = str(urls.index(KAIRYU_CONNECTION_URL))
    configs = dict(config.get("OPENAI_API_CONFIGS") or {})
    if (configs.get(index) or {}).get("api_type") == "responses":
        return
    # A volume from before the move keeps its saved chat connection.
    configs[index] = {**(configs.get(index) or {}), "api_type": "responses"}
    payload = {**config, "OPENAI_API_CONFIGS": configs}
    updated = api(ui_url, "/openai/config/update", token=token, payload=payload)
    if ((updated.get("OPENAI_API_CONFIGS") or {}).get(index) or {}).get("api_type") != "responses":
        raise SystemExit("Open WebUI's Kairyu connection could not be switched to Responses")


def _install_filter(ui_url: str, token: str, workspace: Workspace) -> None:
    """The one global Responses filter; the effort dropdown when the workspace has levels."""
    source = (PLUGINS / "responses_filter.py").read_text(encoding="utf-8")
    base = f"/api/v1/functions/id/{FILTER_ID}"
    listed = api(ui_url, "/api/v1/functions/", token=token)
    for row in listed:
        if row.get("id") in RETIRED_FILTER_IDS:
            api(ui_url, f"/api/v1/functions/id/{row['id']}/delete", token=token, method="DELETE")
    body = {
        "id": FILTER_ID,
        "name": "Reasoning Effort" if workspace.effort_levels else "Responses API",
        "content": source,
        "meta": {"description": "Keep only OpenAI Responses fields in Chat UI requests."},
    }
    existing = next((row for row in listed if row.get("id") == FILTER_ID), None)
    if existing is None:
        state = api(ui_url, "/api/v1/functions/create", token=token, payload=body)
    else:
        state = {**existing, **(api(ui_url, f"{base}/update", token=token, payload=body) or {})}
    # The toggle endpoints flip state, so call them only while a flag is off.
    if not state.get("is_active"):
        state = api(ui_url, f"{base}/toggle", token=token, payload={})
    if not state.get("is_global"):
        state = api(ui_url, f"{base}/toggle/global", token=token, payload={})
    if not (state.get("is_active") and state.get("is_global")):
        raise SystemExit("Open WebUI Responses filter could not be activated globally")
    exported = api(ui_url, "/api/v1/functions/export", token=token)
    installed = next((row for row in exported if row.get("id") == FILTER_ID), {})
    if installed.get("content") != source:
        raise SystemExit("Open WebUI Responses filter is not installed with L4's source")
    if any(row.get("id") in RETIRED_FILTER_IDS for row in exported):
        raise SystemExit("an earlier Responses filter is still installed")
    spec = api(ui_url, f"{base}/valves/user/spec", token=token) or {}
    enum = ((spec.get("properties") or {}).get("reasoning_effort") or {}).get("enum")
    if (enum or []) != list(workspace.effort_levels):
        raise SystemExit(
            f"Chat UI effort selector offers {enum!r}, not {list(workspace.effort_levels)!r}"
        )


def _install_tool(ui_url: str, token: str, tool_id: str) -> None:
    source = TOOLS[tool_id].read_text(encoding="utf-8")
    name, description = _TOOL_META[tool_id]
    body = {"id": tool_id, "name": name, "content": source, "meta": {"description": description}}
    listed = api(ui_url, "/api/v1/tools/", token=token)
    if any(row.get("id") == tool_id for row in listed):
        api(ui_url, f"/api/v1/tools/id/{tool_id}/update", token=token, payload=body)
    else:
        api(ui_url, "/api/v1/tools/create", token=token, payload=body)
    api(
        ui_url,
        f"/api/v1/tools/id/{tool_id}/access/update",
        token=token,
        payload={"access_grants": [USERS_READ]},
    )
    exported = api(ui_url, "/api/v1/tools/export", token=token)
    installed = next((row for row in exported if row.get("id") == tool_id), {})
    if installed.get("content") != source or not installed.get("specs"):
        raise SystemExit(f"Open WebUI tool {tool_id} is not installed with L4's source")
    grants = [
        {key: grant.get(key) for key in USERS_READ}
        for grant in installed.get("access_grants") or ()
    ]
    if USERS_READ not in grants:
        raise SystemExit(f"Open WebUI tool {tool_id} is not readable by every user: {grants}")


def _check_retrieval(ui_url: str, token: str, workspace: Workspace) -> None:
    """Docling extraction and the pinned models. BM25 stays at weight 0: Open
    WebUI splits BM25 terms on whitespace, which cannot tokenize Japanese."""
    retrieval = workspace.profile["retrieval"]
    config = api(ui_url, "/api/v1/retrieval/config", token=token)
    embedding = api(ui_url, "/api/v1/retrieval/embedding", token=token)
    # The embedding settings have their own route; everything else is config.
    observed = {**config, **{key: embedding.get(key) for key in _EMBEDDING_KEYS}}
    expected = {
        "CONTENT_EXTRACTION_ENGINE": "docling",
        "DOCLING_SERVER_URL": DOCLING_URL,
        "DOCLING_PARAMS": workspace.profile["docling"]["params"],
        "RAG_EMBEDDING_ENGINE": "",
        "RAG_EMBEDDING_MODEL": model_path(workspace, "embedding"),
        "RAG_RERANKING_ENGINE": "",
        "RAG_RERANKING_MODEL": model_path(workspace, "reranking"),
        "ENABLE_RAG_HYBRID_SEARCH": True,
        "HYBRID_BM25_WEIGHT": retrieval["hybrid_bm25_weight"],
        "TOP_K": retrieval["top_k"],
        "TOP_K_RERANKER": retrieval["top_k_reranker"],
        "CHUNK_SIZE": retrieval["chunk_size"],
        "CHUNK_OVERLAP": retrieval["chunk_overlap"],
    }
    wrong = _differences({key: observed.get(key) for key in expected}, expected)
    if wrong:
        raise SystemExit(f"Chat UI retrieval differs from the workspace (seen, expected): {wrong}")


def _check_code_interpreter(ui_url: str, token: str, workspace: Workspace) -> None:
    """Word / Excel / PowerPoint need the browser code interpreter, on by default."""
    interpreter = workspace.profile["code_interpreter"]
    config = api(ui_url, "/api/config", token=token)
    features, code = config.get("features") or {}, config.get("code") or {}
    expected_meta = {
        "defaultFeatureIds": sorted(interpreter["default_features"]),
        "toolIds": sorted(interpreter["default_tools"]),
        "capabilities": DEFAULT_CAPABILITIES,
    }
    expected = {
        "enable_code_interpreter": True,
        "interpreter_engine": interpreter["engine"],
        "enable_pyodide_file_persistence": interpreter["file_persistence"],
        **{model: expected_meta for model in workspace.models},
    }
    models = {row.get("id"): row for row in api(ui_url, "/api/models", token=token)["data"]}
    observed = {
        "enable_code_interpreter": features.get("enable_code_interpreter"),
        "interpreter_engine": code.get("interpreter_engine"),
        "enable_pyodide_file_persistence": features.get("enable_pyodide_file_persistence"),
    }
    for model in workspace.models:
        meta = ((models.get(model) or {}).get("info") or {}).get("meta") or {}
        observed[model] = {
            "defaultFeatureIds": sorted(meta.get("defaultFeatureIds") or []),
            "toolIds": sorted(meta.get("toolIds") or []),
            "capabilities": meta.get("capabilities"),
        }
    wrong = _differences(observed, expected)
    if wrong:
        raise SystemExit(f"Chat UI code interpreter differs (seen, expected): {wrong}")


def setup(ui_url: str, workspace: Workspace, *, email: str = "", password: str = "") -> None:
    """Install L4's plugins in a running Open WebUI and check its configuration."""
    try:
        token = sign_in(ui_url, workspace.auth, email, password)
        version = api(ui_url, "/api/version").get("version")
        if version != OPEN_WEBUI_VERSION:
            raise SystemExit(f"Open WebUI {version} is running; L4 supports {OPEN_WEBUI_VERSION}")
        _ensure_responses_connection(ui_url, token)
        _install_filter(ui_url, token, workspace)
        offered = {row.get("id") for row in api(ui_url, "/api/models", token=token)["data"]}
        missing = sorted(set(workspace.models) - offered)
        if missing:
            raise SystemExit(f"Open WebUI does not offer {missing}; it offers {sorted(offered)}")
        if workspace.profile is not None:
            _check_retrieval(ui_url, token, workspace)
            for tool_id in workspace.profile["code_interpreter"]["default_tools"]:
                _install_tool(ui_url, token, tool_id)
            _check_code_interpreter(ui_url, token, workspace)
    except (KeyError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Open WebUI setup failed: {error!r}") from error
