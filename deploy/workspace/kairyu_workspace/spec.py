"""An L4 workspace: an example's ``workspace`` section plus the profile it names.

The section is checked when it is loaded, so an example's ``control.py``
refuses a wrong one at import time. Standard library only: example launchers
run on the host's Python.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "compose.yaml"
PROFILES = ROOT / "profiles"
PLUGINS = ROOT / "plugins"

# The one Open WebUI L4 supports: the plugins use its internals, and setup
# checks /api/version against it. Moving to another release is one change here.
OPEN_WEBUI_VERSION = "0.12.3"
OPEN_WEBUI_IMAGE = (
    "ghcr.io/open-webui/open-webui:v0.12.3"
    "@sha256:ef11a68f38457d831e23168486858c25b9d9c4399094c9e72c0d4d8767ac09de"
)
# Open WebUI v0.12.3's own DEFAULT_CAPABILITIES. A default feature applies only
# to a model whose capabilities include it, so the model defaults carry this
# set unchanged.
DEFAULT_CAPABILITIES = {
    "file_context": True,
    "vision": True,
    "file_upload": True,
    "web_search": True,
    "image_generation": True,
    "code_interpreter": True,
    "terminal": True,
    "citations": True,
    "status_updates": True,
    "memory": True,
    "builtin_tools": True,
}
# The Chat UI reads the retrieval models from this read-only mount.
MODELS_MOUNT = "/opt/kairyu-workspace-models"
DOCLING_URL = "http://docling:5001"
# Tools a profile may turn on by default, by Open WebUI tool id.
TOOLS = {"file_download": PLUGINS / "file_download_tool.py"}
# Every file that decides what a workspace serves (evidence hashes).
SERVED_FILES = (
    COMPOSE_FILE,
    PLUGINS / "responses_filter.py",
    *TOOLS.values(),
    *sorted(PROFILES.glob("*.json")),
    *sorted(Path(__file__).parent.glob("*.py")),
)

_SETUP_KEYS = {"profile", "auth", "models", "reasoning_effort"}
_DEPLOY_KEYS = {"port", "bind", "default_params", "cpus"}
_CPU_KEYS = {"chat_ui", "docling", "docling_workers"}
_PROFILE_KEYS = {"name", "docling", "retrieval", "code_interpreter", "gates"}
_RETRIEVAL_KEYS = {
    "embedding",
    "reranking",
    "chunk_size",
    "chunk_overlap",
    "hybrid_bm25_weight",
    "top_k",
    "top_k_reranker",
}
_MODEL_KEYS = {"repo", "revision", "slug", "tree_sha256", "allow_patterns"}


@dataclass(frozen=True)
class Workspace:
    """A checked workspace section. ``profile`` None means setup only: the
    Open WebUI is deployed elsewhere (the CPU demo) and L4 only installs its
    filter and checks its connection."""

    profile: dict | None
    auth: str
    models: tuple[str, ...]
    effort_levels: tuple[str, ...]
    effort_help: str
    port: int | None = None
    bind: str | None = None
    default_params: dict | None = None
    cpus: dict | None = None


def _keys(where: str, value: object, required: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    missing, unknown = required - set(value), set(value) - required
    if missing or unknown:
        raise ValueError(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    return value


def load_profile(name: str) -> dict:
    """A named, immutable profile: a change is a new name, never an edit."""
    path = PROFILES / f"{name}.json"
    if not isinstance(name, str) or not path.is_file():
        raise ValueError(f"unknown workspace profile {name!r}")
    profile = _keys(f"profile {name}", json.loads(path.read_text(encoding="utf-8")), _PROFILE_KEYS)
    if profile["name"] != name:
        raise ValueError(f"profile {path.name} is named {profile['name']!r}")
    retrieval = _keys(f"profile {name} retrieval", profile["retrieval"], _RETRIEVAL_KEYS)
    for role in ("embedding", "reranking"):
        _keys(f"profile {name} {role}", retrieval[role], _MODEL_KEYS)
    unknown = set(profile["code_interpreter"]["default_tools"]) - set(TOOLS)
    if unknown:
        raise ValueError(f"profile {name} names tools L4 does not ship: {sorted(unknown)}")
    return profile


def load(section: dict) -> Workspace:
    """Check an example's ``workspace`` section and resolve its profile."""
    profile_name = section.get("profile") if isinstance(section, dict) else None
    deployed = profile_name is not None
    _keys("workspace", section, _SETUP_KEYS | (_DEPLOY_KEYS if deployed else set()))
    if section["auth"] not in ("none", "admin"):
        raise ValueError("workspace auth must be 'none' or 'admin'")
    models = section["models"]
    if not models or not all(isinstance(model, str) and model for model in models):
        raise ValueError("workspace models must name at least one served model")
    effort = section["reasoning_effort"]
    levels: tuple[str, ...] = ()
    help_text = ""
    if effort is not None:
        _keys("workspace reasoning_effort", effort, {"levels", "help"})
        levels = tuple(effort["levels"])
        help_text = effort["help"]
        if not levels or levels[0] != "default" or len(set(levels)) != len(levels):
            raise ValueError("reasoning_effort levels must be unique and start with 'default'")
    if not deployed:
        return Workspace(None, section["auth"], tuple(models), levels, help_text)
    cpus = _keys("workspace cpus", section["cpus"], _CPU_KEYS)
    if not all(isinstance(count, int) and count > 0 for count in cpus.values()):
        raise ValueError("workspace cpus must be positive integers")
    if cpus["docling"] % cpus["docling_workers"]:
        raise ValueError("workspace cpus.docling must divide evenly among its workers")
    if not isinstance(section["port"], int) or not isinstance(section["default_params"], dict):
        raise ValueError("workspace port must be an integer and default_params an object")
    return Workspace(
        load_profile(profile_name),
        section["auth"],
        tuple(models),
        levels,
        help_text,
        port=section["port"],
        bind=section["bind"],
        default_params=section["default_params"],
        cpus=dict(cpus),
    )


def model_path(workspace: Workspace, role: str) -> str:
    """Where the Chat UI finds a retrieval model ("embedding" or "reranking")."""
    return f"{MODELS_MOUNT}/{workspace.profile['retrieval'][role]['slug']}"


def chat_ui_port(workspace: Workspace) -> str:
    return os.environ.get("CHAT_UI_PORT", str(workspace.port))


def chat_ui_bind(workspace: Workspace) -> str:
    return os.environ.get("CHAT_UI_BIND_ADDRESS", workspace.bind)


def compose_env(workspace: Workspace, *, data: Path, models: Path) -> dict[str, str]:
    """The variables ``compose.yaml`` reads; ``data`` and ``models`` are host paths."""
    if workspace.profile is None:
        raise ValueError("a deployed workspace needs a profile")
    profile = workspace.profile
    retrieval = profile["retrieval"]
    interpreter = profile["code_interpreter"]
    cpus = workspace.cpus
    port = chat_ui_port(workspace)
    admin = workspace.auth == "admin"
    return {
        "KAIRYU_WORKSPACE_OPEN_WEBUI_IMAGE": OPEN_WEBUI_IMAGE,
        "KAIRYU_WORKSPACE_DOCLING_IMAGE": profile["docling"]["image"],
        "KAIRYU_WORKSPACE_DATA": str(data),
        "KAIRYU_WORKSPACE_MODELS": str(models),
        "KAIRYU_WORKSPACE_BIND": chat_ui_bind(workspace),
        "KAIRYU_WORKSPACE_PORT": port,
        "KAIRYU_WORKSPACE_URL": os.environ.get("WEBUI_URL", f"http://localhost:{port}"),
        "KAIRYU_WORKSPACE_AUTH": str(admin).lower(),
        "KAIRYU_WORKSPACE_DEFAULT_MODEL": workspace.models[0],
        "KAIRYU_WORKSPACE_DEFAULT_MODEL_PARAMS": json.dumps(
            workspace.default_params, separators=(",", ":")
        ),
        "KAIRYU_WORKSPACE_DEFAULT_MODEL_METADATA": json.dumps(
            {
                "capabilities": DEFAULT_CAPABILITIES,
                "defaultFeatureIds": interpreter["default_features"],
                "toolIds": interpreter["default_tools"],
            },
            separators=(",", ":"),
        ),
        "KAIRYU_WORKSPACE_EFFORT_LEVELS": json.dumps(list(workspace.effort_levels)),
        "KAIRYU_WORKSPACE_EFFORT_HELP": workspace.effort_help,
        "KAIRYU_WORKSPACE_DOCLING_PARAMS": json.dumps(
            profile["docling"]["params"], separators=(",", ":")
        ),
        "KAIRYU_WORKSPACE_EMBEDDING_MODEL": model_path(workspace, "embedding"),
        "KAIRYU_WORKSPACE_RERANKING_MODEL": model_path(workspace, "reranking"),
        "KAIRYU_WORKSPACE_BM25_WEIGHT": str(retrieval["hybrid_bm25_weight"]),
        "KAIRYU_WORKSPACE_TOP_K": str(retrieval["top_k"]),
        "KAIRYU_WORKSPACE_TOP_K_RERANKER": str(retrieval["top_k_reranker"]),
        "KAIRYU_WORKSPACE_CHUNK_SIZE": str(retrieval["chunk_size"]),
        "KAIRYU_WORKSPACE_CHUNK_OVERLAP": str(retrieval["chunk_overlap"]),
        "KAIRYU_WORKSPACE_CODE_INTERPRETER_ENGINE": interpreter["engine"],
        "KAIRYU_WORKSPACE_PYODIDE_FILE_PERSISTENCE": str(interpreter["file_persistence"]).lower(),
        "KAIRYU_WORKSPACE_CHAT_UI_CPUS": str(cpus["chat_ui"]),
        "KAIRYU_WORKSPACE_DOCLING_CPUS": str(cpus["docling"]),
        "KAIRYU_WORKSPACE_DOCLING_WORKERS": str(cpus["docling_workers"]),
        "KAIRYU_WORKSPACE_DOCLING_THREADS": str(cpus["docling"] // cpus["docling_workers"]),
        "WEBUI_ADMIN_EMAIL": os.environ.get("WEBUI_ADMIN_EMAIL", "") if admin else "",
        "WEBUI_ADMIN_PASSWORD": os.environ.get("WEBUI_ADMIN_PASSWORD", "") if admin else "",
    }
