#!/usr/bin/env python3
"""One-command lifecycle for the DeepSeek-V4-Flash-0731 eight-GPU example."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
ROOT = HERE.parents[1]


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(command[:4])
    print(f"+ {printable}{' ...' if len(command) > 4 else ''}", flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _compose_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME"}
    }
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": SPEC["environment"].replace(".", "-"),
            "MODEL_VOLUME": os.environ.get("MODEL_VOLUME", SPEC["model_volume"]),
            "VLLM_IMAGE": os.environ.get("VLLM_IMAGE", SPEC["vllm"]["image"]),
            "OPEN_WEBUI_IMAGE": os.environ.get(
                "OPEN_WEBUI_IMAGE", SPEC["webui"]["image"]
            ),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "CHAT_UI_PORT": os.environ.get(
                "CHAT_UI_PORT", str(SPEC["webui"]["port"])
            ),
            "CHAT_UI_BIND_ADDRESS": os.environ.get(
                "CHAT_UI_BIND_ADDRESS", "127.0.0.1"
            ),
        }
    )
    return env


def _compose(arguments: list[str], *, check: bool = True) -> None:
    _run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(HERE),
            "--file",
            str(HERE / "compose.yaml"),
            *arguments,
        ],
        env=_compose_env(),
        check=check,
    )


def _gpu_inventory(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = [part.strip() for part in raw.split(",")]
        if len(fields) != 4:
            raise SystemExit(f"unexpected nvidia-smi row: {raw}")
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_mib": int(fields[2]),
                "compute_capability": float(fields[3]),
            }
        )
    return rows


def _preflight() -> None:
    for executable in ("docker", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable is missing: {executable}")
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).stdout
    rows = _gpu_inventory(query)
    expected = SPEC["hardware"]
    if len(rows) != expected["gpu_count"]:
        raise SystemExit(
            f"exactly {expected['gpu_count']} GPUs are required; found {len(rows)}"
        )
    for position, row in enumerate(rows):
        if row["index"] != position:
            raise SystemExit("GPU indices must be contiguous 0..7")
        if row["name"] != expected["product"]:
            raise SystemExit(
                f"GPU {position} is {row['name']!r}; expected {expected['product']!r}"
            )
        if row["memory_mib"] < expected["minimum_vram_mib"]:
            raise SystemExit(f"GPU {position} has insufficient VRAM")
        if row["compute_capability"] < expected["minimum_compute_capability"]:
            raise SystemExit(f"GPU {position} has insufficient compute capability")
    print(
        f"hardware: {len(rows)} x {expected['product']} "
        f"({rows[0]['memory_mib']} MiB each)",
        flush=True,
    )


def _image_exists(image: str) -> bool:
    return (
        _run(
            ["docker", "image", "inspect", image],
            capture=True,
            check=False,
        ).returncode
        == 0
    )


def _image_id(image: str) -> str | None:
    inspected = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture=True,
        check=False,
    )
    return inspected.stdout.strip() if inspected.returncode == 0 else None


def _ensure_vllm_image() -> None:
    """Build the pinned SM120 source revision and the effort overlay, then attest it."""

    image = _compose_env()["VLLM_IMAGE"]
    source = SPEC["vllm"]
    if not _image_exists(image):
        _build_vllm_image(image)
    actual = _image_id(image)
    if os.environ.get("VLLM_ALLOW_UNPINNED_IMAGE") == "1":
        print(f"WARNING: serving unpinned image {image} ({actual}); not evidence", flush=True)
        return
    if source["image_id"] is None:
        raise SystemExit(
            f"vLLM image {image} ({actual}) has no recorded image ID: it was rebuilt for "
            "the Responses xhigh->max effort edit and awaits GPU verification; record this "
            "ID as example.json image_id and kairyu.yaml container_image_digest, or serve it "
            "anyway (not evidence) with VLLM_ALLOW_UNPINNED_IMAGE=1"
        )
    if actual != source["image_id"]:
        raise SystemExit(
            f"vLLM image {image} has ID {actual}; example.json and kairyu.yaml pin "
            f"{source['image_id']} (update both to this build before serving)"
        )


def _build_vllm_image(image: str) -> None:
    source = SPEC["vllm"]
    if image != source["image"]:
        raise SystemExit(f"VLLM_IMAGE does not exist locally: {image}")
    base = source["base_image"]
    if not _image_exists(base):
        print("vLLM base image is absent; building the pinned SM120 source revision", flush=True)
        _run(
            [
                "docker",
                "build",
                "--pull",
                "--file",
                "docker/Dockerfile",
                "--tag",
                base,
                "--label",
                f"org.opencontainers.image.revision={source['source_revision']}",
                f"{source['source_repository']}#{source['source_revision']}",
            ]
        )
    print("vLLM image is absent; building this example's effort overlay", flush=True)
    _run(
        [
            "docker",
            "build",
            "--file",
            str(HERE / source["dockerfile"]),
            "--build-arg",
            f"VLLM_BASE_IMAGE={base}",
            "--tag",
            image,
            "--label",
            f"org.opencontainers.image.revision={source['source_revision']}",
            str(HERE),
        ]
    )


_MODEL_PROGRAM = r"""
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, revision, slug, expected_tree = sys.argv[1:]
target = Path('/models') / slug
attestation = target / '.kairyu-model-attestation.json'

def inventory():
    rows = []
    for path in sorted(target.rglob('*')):
        if not path.is_file() or path == attestation or '.cache' in path.parts:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                digest.update(chunk)
        rows.append({'path': str(path.relative_to(target)), 'size': path.stat().st_size,
                     'sha256': digest.hexdigest()})
    tree = hashlib.sha256(json.dumps(rows, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()
    return rows, tree

if attestation.exists():
    current = json.loads(attestation.read_text())
    fields = (current.get('repo'), current.get('revision'), current.get('tree_sha256'))
    if fields != (repo, revision, expected_tree):
        raise SystemExit('model attestation does not match the pinned checkpoint')
    if os.environ.get('VERIFY_MODEL') != '1':
        raise SystemExit(0)
    rows, tree = inventory()
    if tree != expected_tree or current.get('files') != rows:
        raise SystemExit('model files differ from the pinned attestation')
    raise SystemExit(0)

snapshot_download(repo, revision=revision, local_dir=target,
                  token=os.environ.get('HF_TOKEN') or None)
rows, tree = inventory()
if tree != expected_tree:
    raise SystemExit(f'downloaded checkpoint tree mismatch: {tree}')
attestation.write_text(json.dumps({'schema_version': 1, 'repo': repo,
                                    'revision': revision, 'tree_sha256': tree,
                                    'files': rows}, sort_keys=True))
"""


def _ensure_model() -> None:
    env = _compose_env()
    volume = env["MODEL_VOLUME"]
    _run(["docker", "volume", "create", volume], capture=True)
    model = SPEC["model"]
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        "--volume",
        f"{volume}:/models",
    ]
    if "HF_TOKEN" in os.environ:
        command.extend(["--env", "HF_TOKEN"])
    if os.environ.get("VERIFY_MODEL") == "1":
        command.extend(["--env", "VERIFY_MODEL=1"])
    command.extend(
        [
            env["VLLM_IMAGE"],
            "-c",
            _MODEL_PROGRAM,
            model["repo"],
            model["revision"],
            model["slug"],
            model["tree_sha256"],
        ]
    )
    _run(command)


def _ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


# Owner decision 2026-10-17: the Chat UI talks to Kairyu over /v1/responses;
# with WEBUI_ADMIN_EMAIL/WEBUI_ADMIN_PASSWORD set, `up` installs the Responses
# filter and checks the connection type, otherwise it warns.
_CHAT_UI_FILTER_ID = "responses_api"


def _webui_api(
    ui_url: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict | None = None,
):
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


_KAIRYU_CONNECTION_URL = "http://kairyu:8000/v1"


def _ensure_chat_ui_connection(ui_url: str, token: str) -> None:
    """Point the Chat UI's Kairyu connection at /v1/responses (m11 D7 amendment).

    Open WebUI keeps the connection a volume saved before this example moved
    to the Responses API; owner decision 2026-10-17: the admin session
    rewrites that one connection's API type and leaves everything else.
    """

    config = _webui_api(ui_url, "/openai/config", token=token)
    urls = list(config.get("OPENAI_API_BASE_URLS") or [])
    if _KAIRYU_CONNECTION_URL not in urls:
        raise SystemExit(f"Chat UI has no connection to {_KAIRYU_CONNECTION_URL}: {urls!r}")
    index = str(urls.index(_KAIRYU_CONNECTION_URL))
    configs = dict(config.get("OPENAI_API_CONFIGS") or {})
    if (configs.get(index) or {}).get("api_type") == "responses":
        return
    configs[index] = {**(configs.get(index) or {}), "api_type": "responses"}
    updated = _webui_api(
        ui_url,
        "/openai/config/update",
        token=token,
        payload={**config, "OPENAI_API_CONFIGS": configs},
    )
    if ((updated.get("OPENAI_API_CONFIGS") or {}).get(index) or {}).get("api_type") != "responses":
        raise SystemExit("Chat UI Kairyu connection could not be switched to api_type=responses")
    print("Chat UI: switched the saved Kairyu connection to the Responses API", flush=True)


def _provision_chat_ui(ui_url: str) -> None:
    """Install the Responses filter and check the connection type (admin only)."""

    email = os.environ.get("WEBUI_ADMIN_EMAIL", "")
    password = os.environ.get("WEBUI_ADMIN_PASSWORD", "")
    if not (email and password):
        print(
            "warning: WEBUI_ADMIN_EMAIL/WEBUI_ADMIN_PASSWORD are unset, so the Chat UI's "
            "Responses filter was not installed: Chat Controls fields outside the Responses "
            "API (reasoning effort, seed, top_k) are rejected until an admin installs "
            "webui-responses-filter.py as a global filter (README, Chat UI)",
            file=sys.stderr,
        )
        return
    filter_source = (HERE / "webui-responses-filter.py").read_text(encoding="utf-8")
    base = f"/api/v1/functions/id/{_CHAT_UI_FILTER_ID}"
    try:
        signin = _webui_api(
            ui_url, "/api/v1/auths/signin", payload={"email": email, "password": password}
        )
        if not isinstance(signin, dict) or not signin.get("token"):
            raise SystemExit("Chat UI admin signin did not return a session token")
        if signin.get("role") != "admin":
            raise SystemExit("WEBUI_ADMIN_EMAIL must be the Chat UI's admin user")
        token = signin["token"]
        listed = _webui_api(ui_url, "/api/v1/functions/", token=token)
        existing = next((row for row in listed if row.get("id") == _CHAT_UI_FILTER_ID), None)
        body = {
            "id": _CHAT_UI_FILTER_ID,
            "name": "Responses API",
            "content": filter_source,
            "meta": {"description": "Keep only OpenAI Responses fields in Chat UI requests."},
        }
        if existing is None:
            state = _webui_api(ui_url, "/api/v1/functions/create", token=token, payload=body)
        else:
            updated = _webui_api(ui_url, f"{base}/update", token=token, payload=body)
            state = {**existing, **(updated or {})}
        # The toggle endpoints flip state, so call them only while a flag is off.
        if not state.get("is_active"):
            state = _webui_api(ui_url, f"{base}/toggle", token=token, payload={})
        if not state.get("is_global"):
            state = _webui_api(ui_url, f"{base}/toggle/global", token=token, payload={})
        if not (state.get("is_active") and state.get("is_global")):
            raise SystemExit("Chat UI Responses filter could not be activated globally")
        _ensure_chat_ui_connection(ui_url, token)
    except (KeyError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Chat UI provisioning failed: {error}") from error


def up() -> None:
    _preflight()
    _ensure_vllm_image()
    _ensure_model()
    _compose(["up", "--build", "--detach", "--wait", "--wait-timeout", "5400"])
    env = _compose_env()
    api_url = f"http://127.0.0.1:{env['API_PORT']}"
    if not _ready(f"{api_url}/readyz"):
        raise SystemExit("Kairyu did not become ready")
    ui_bind = env["CHAT_UI_BIND_ADDRESS"]
    _provision_chat_ui(
        f"http://{'127.0.0.1' if ui_bind == '0.0.0.0' else ui_bind}:{env['CHAT_UI_PORT']}"
    )
    print("\nEnvironment is ready.")
    print(f"OpenAI API: {api_url}/v1")
    ui_host = os.environ.get("PUBLIC_HOST", env["CHAT_UI_BIND_ADDRESS"])
    if ui_host == "0.0.0.0":
        ui_host = "127.0.0.1"
    print(f"Chat UI:    http://{ui_host}:{env['CHAT_UI_PORT']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("up", "down", "status", "logs"), default="up")
    args = parser.parse_args()
    if args.action == "up":
        up()
    elif args.action == "down":
        _compose(["down"])
    elif args.action == "status":
        _compose(["ps"])
    else:
        _compose(["logs", "--follow", "--tail", "200"])


if __name__ == "__main__":
    main()
