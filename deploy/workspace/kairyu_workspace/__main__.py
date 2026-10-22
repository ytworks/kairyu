"""``python3 -m kairyu_workspace setup``: set up an Open WebUI deployed elsewhere.

The CPU demo (``deploy/compose/docker-compose.webui.yaml``) runs this in the
pinned Open WebUI image with its workspace section in ``KAIRYU_WORKSPACE_SPEC``.
"""

from __future__ import annotations

import argparse
import json
import os

from . import load, setup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup",))
    parser.parse_args()
    workspace = load(json.loads(os.environ["KAIRYU_WORKSPACE_SPEC"]))
    setup(
        os.environ.get("WEBUI_URL", "http://webui:8080").rstrip("/"),
        workspace,
        email=os.environ.get("WEBUI_ADMIN_EMAIL", ""),
        password=os.environ.get("WEBUI_ADMIN_PASSWORD", ""),
    )
    print("Open WebUI: Responses filter installed and active globally", flush=True)


if __name__ == "__main__":
    main()
