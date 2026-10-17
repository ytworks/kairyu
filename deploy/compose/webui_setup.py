"""Install the compose demo's Responses filter in Open WebUI (m11 D7 amendment).

The demo's Open WebUI talks to Kairyu over /v1/responses, and Kairyu rejects
fields outside the Responses API (Chat Controls effort, seed, top_k; saved
chat-era output fields). With WEBUI_ADMIN_EMAIL/WEBUI_ADMIN_PASSWORD set, this
one-shot step signs in as that admin, installs webui-responses-filter.py as an
active global filter, and switches a connection saved before the move to
Responses; without them it only warns. Stdlib only: it runs in the Open WebUI
image.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

FILTER_ID = "responses_api"
FILTER_SOURCE = Path(__file__).with_name("webui-responses-filter.py")
KAIRYU_CONNECTION_URL = "http://kairyu:8000/v1"


def _api(ui_url: str, path: str, *, token: str | None = None, payload: dict | None = None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{ui_url}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _install_filter(ui_url: str, token: str) -> None:
    base = f"/api/v1/functions/id/{FILTER_ID}"
    listed = _api(ui_url, "/api/v1/functions/", token=token)
    existing = next((row for row in listed if row.get("id") == FILTER_ID), None)
    body = {
        "id": FILTER_ID,
        "name": "Responses API",
        "content": FILTER_SOURCE.read_text(encoding="utf-8"),
        "meta": {"description": "Keep only OpenAI Responses fields in Chat UI requests."},
    }
    if existing is None:
        state = _api(ui_url, "/api/v1/functions/create", token=token, payload=body)
    else:
        state = {**existing, **(_api(ui_url, f"{base}/update", token=token, payload=body) or {})}
    # The toggle endpoints flip state, so call them only while a flag is off.
    if not state.get("is_active"):
        state = _api(ui_url, f"{base}/toggle", token=token, payload={})
    if not state.get("is_global"):
        state = _api(ui_url, f"{base}/toggle/global", token=token, payload={})
    if not (state.get("is_active") and state.get("is_global")):
        raise SystemExit("Open WebUI Responses filter could not be activated globally")


def _ensure_responses_connection(ui_url: str, token: str) -> None:
    config = _api(ui_url, "/openai/config", token=token)
    urls = list(config.get("OPENAI_API_BASE_URLS") or [])
    if KAIRYU_CONNECTION_URL not in urls:
        raise SystemExit(f"Open WebUI has no connection to {KAIRYU_CONNECTION_URL}: {urls!r}")
    index = str(urls.index(KAIRYU_CONNECTION_URL))
    configs = dict(config.get("OPENAI_API_CONFIGS") or {})
    if (configs.get(index) or {}).get("api_type") == "responses":
        return
    configs[index] = {**(configs.get(index) or {}), "api_type": "responses"}
    updated = _api(
        ui_url,
        "/openai/config/update",
        token=token,
        payload={**config, "OPENAI_API_CONFIGS": configs},
    )
    if ((updated.get("OPENAI_API_CONFIGS") or {}).get(index) or {}).get("api_type") != "responses":
        raise SystemExit("Open WebUI's Kairyu connection could not be switched to Responses")
    print("Open WebUI: switched the saved Kairyu connection to the Responses API", flush=True)


def main() -> None:
    ui_url = os.environ.get("WEBUI_URL", "http://webui:8080").rstrip("/")
    email = os.environ.get("WEBUI_ADMIN_EMAIL", "")
    password = os.environ.get("WEBUI_ADMIN_PASSWORD", "")
    if not (email and password):
        print(
            "warning: WEBUI_ADMIN_EMAIL/WEBUI_ADMIN_PASSWORD are unset, so the Responses "
            "filter was not installed: Chat Controls fields outside the Responses API "
            "(reasoning effort, seed, top_k) are rejected until an admin installs "
            "deploy/compose/webui-responses-filter.py as a global filter",
            file=sys.stderr,
        )
        return
    try:
        signin = _api(
            ui_url, "/api/v1/auths/signin", payload={"email": email, "password": password}
        )
        if not isinstance(signin, dict) or not signin.get("token"):
            raise SystemExit("Open WebUI admin sign-in did not return a session token")
        if signin.get("role") != "admin":
            raise SystemExit("WEBUI_ADMIN_EMAIL must be Open WebUI's admin user")
        _install_filter(ui_url, signin["token"])
        _ensure_responses_connection(ui_url, signin["token"])
    except (KeyError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Open WebUI setup failed: {error}") from error
    print("Open WebUI: Responses filter installed and active globally", flush=True)


if __name__ == "__main__":
    main()
