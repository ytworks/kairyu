#!/usr/bin/env python3
"""One-command lifecycle for the DeepSeek-V4-Flash-0731 eight-GPU example."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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


def _ensure_vllm_image() -> None:
    image = _compose_env()["VLLM_IMAGE"]
    if _image_exists(image):
        return
    source = SPEC["vllm"]
    if image != source["image"]:
        raise SystemExit(f"VLLM_IMAGE does not exist locally: {image}")
    print("vLLM image is absent; building the pinned SM120 source revision", flush=True)
    _run(
        [
            "docker",
            "build",
            "--pull",
            "--build-arg",
            "BUILDKIT_CONTEXT_KEEP_GIT_DIR=1",
            "--file",
            "docker/Dockerfile",
            "--tag",
            image,
            "--label",
            f"org.opencontainers.image.revision={source['source_revision']}",
            f"{source['source_repository']}#{source['source_revision']}",
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


def up() -> None:
    _preflight()
    _ensure_vllm_image()
    _ensure_model()
    _compose(["up", "--build", "--detach", "--wait", "--wait-timeout", "5400"])
    env = _compose_env()
    api_url = f"http://127.0.0.1:{env['API_PORT']}"
    if not _ready(f"{api_url}/readyz"):
        raise SystemExit("Kairyu did not become ready")
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
