"""L4 chat workspace: Open WebUI and its document services in front of Kairyu's L3.

Examples load their ``workspace`` section with :func:`load`, start
``compose.yaml`` after their own compose file with :func:`compose_env`, fetch
the retrieval models with :func:`fetch_models` and finish with :func:`setup`
(``deploy/workspace/README.md``).
"""

from .models import fetch as fetch_models
from .openwebui import setup
from .spec import (
    COMPOSE_FILE,
    OPEN_WEBUI_IMAGE,
    OPEN_WEBUI_VERSION,
    SERVED_FILES,
    Workspace,
    chat_ui_bind,
    chat_ui_port,
    compose_env,
    load,
)

__all__ = [
    "COMPOSE_FILE",
    "OPEN_WEBUI_IMAGE",
    "OPEN_WEBUI_VERSION",
    "SERVED_FILES",
    "Workspace",
    "chat_ui_bind",
    "chat_ui_port",
    "compose_env",
    "fetch_models",
    "load",
    "setup",
]
