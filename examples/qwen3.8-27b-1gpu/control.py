#!/usr/bin/env python3
"""One-command lifecycle for Qwen3.8-27B on one RTX PRO 6000."""

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


def _storage_paths() -> tuple[Path, Path, Path]:
    configured = Path(os.environ.get("NVME_STORAGE_ROOT", SPEC["storage"]["root"]))
    if not configured.is_absolute():
        raise SystemExit("NVME_STORAGE_ROOT must be an absolute path below /mnt/nvme")
    root = configured.resolve()
    nvme = Path("/mnt/nvme")
    if root != nvme and nvme not in root.parents:
        raise SystemExit("NVME_STORAGE_ROOT must be /mnt/nvme or one of its descendants")
    environment_root = root / "model-volumes" / SPEC["environment"]
    model = environment_root / "models"
    webui = environment_root / "webui-data"
    vllm_cache = environment_root / "vllm-cache"
    for path in (model, webui, vllm_cache):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(f"cannot prepare NVMe storage {path}: {error}") from error
    free_gib = shutil.disk_usage(root).free // (1024**3)
    minimum = int(SPEC["storage"]["minimum_free_gib"])
    if free_gib < minimum:
        raise SystemExit(f"NVMe storage has {free_gib} GiB free; {minimum} GiB is required")
    return model, webui, vllm_cache


def _compose_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME"}
    }
    model_path, webui_path, vllm_cache_path = _storage_paths()
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": SPEC["environment"].replace(".", "-"),
            "MODEL_STORAGE_PATH": str(model_path),
            "WEBUI_STORAGE_PATH": str(webui_path),
            "VLLM_CACHE_PATH": str(vllm_cache_path),
            "VLLM_IMAGE": os.environ.get("VLLM_IMAGE", SPEC["vllm"]["image"]),
            "OPEN_WEBUI_IMAGE": os.environ.get("OPEN_WEBUI_IMAGE", SPEC["webui"]["image"]),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "CHAT_UI_PORT": os.environ.get("CHAT_UI_PORT", str(SPEC["webui"]["port"])),
            "CHAT_UI_BIND_ADDRESS": os.environ.get("CHAT_UI_BIND_ADDRESS", "127.0.0.1"),
            "GPU_ID": os.environ.get("GPU_ID", "0"),
            # `up` replaces this with the selected GPU's NUMA-local CPUs.
            # A harmless valid default lets `down/status/logs` render Compose
            # without rerunning the hardware preflight.
            "GPU_CPUSET": os.environ.get("GPU_CPUSET", "0"),
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


def _gpu_inventory(text: str) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = [part.strip() for part in raw.split(",")]
        if len(fields) != 5:
            raise SystemExit(f"unexpected nvidia-smi row: {raw}")
        index = int(fields[0])
        rows[index] = {
            "name": fields[1],
            "memory_mib": int(fields[2]),
            "compute_capability": float(fields[3]),
            "pci_bus_id": fields[4].lower(),
        }
    return rows


def _numa_cpuset(pci_bus_id: str) -> str:
    canonical = pci_bus_id
    if canonical.startswith("00000000:"):
        canonical = canonical[4:]
    try:
        node = int((Path("/sys/bus/pci/devices") / canonical / "numa_node").read_text())
        if node < 0:
            raise ValueError("negative NUMA node")
        cpuset = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"cannot determine NUMA affinity for GPU {pci_bus_id}: {error}"
        ) from error
    if not cpuset:
        raise SystemExit(f"NUMA CPU set for GPU {pci_bus_id} is empty")
    return cpuset


def _preflight(env: dict[str, str]) -> None:
    for executable in ("docker", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable is missing: {executable}")
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).stdout
    rows = _gpu_inventory(query)
    try:
        selected = int(env["GPU_ID"])
    except ValueError as error:
        raise SystemExit("GPU_ID must be an integer") from error
    if selected not in rows:
        raise SystemExit(f"GPU_ID={selected} is unavailable")
    row = rows[selected]
    expected = SPEC["hardware"]
    if row["name"] != expected["product"]:
        raise SystemExit(f"GPU {selected} is {row['name']!r}; expected {expected['product']!r}")
    if row["memory_mib"] < expected["minimum_vram_mib"]:
        raise SystemExit(f"GPU {selected} has insufficient VRAM")
    if row["compute_capability"] < expected["minimum_compute_capability"]:
        raise SystemExit(f"GPU {selected} has insufficient compute capability")
    env["GPU_CPUSET"] = _numa_cpuset(str(row["pci_bus_id"]))
    print(
        f"hardware: GPU {selected}, {row['name']} ({row['memory_mib']} MiB), "
        f"cpuset {env['GPU_CPUSET']}",
        flush=True,
    )


def _image_exists(image: str) -> bool:
    return _run(["docker", "image", "inspect", image], capture=True, check=False).returncode == 0


def _ensure_vllm_image(env: dict[str, str]) -> None:
    image = env["VLLM_IMAGE"]
    if _image_exists(image):
        return
    source = SPEC["vllm"]
    if image != source["image"]:
        raise SystemExit(f"VLLM_IMAGE does not exist locally: {image}")
    if source.get("distribution") == "upstream":
        print("vLLM image is absent; pulling the pinned upstream release", flush=True)
        _run(["docker", "pull", image])
        return
    print("vLLM image is absent; building the pinned SM120 source revision", flush=True)
    _run(
        [
            "docker",
            "build",
            "--pull",
            "--file",
            "docker/Dockerfile",
            "--tag",
            image,
            "--label",
            f"org.opencontainers.image.revision={source['source_revision']}",
            f"{source['source_repository']}#{source.get('source_ref', source['source_revision'])}",
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


def _ensure_model(env: dict[str, str]) -> None:
    model = SPEC["model"]
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        "--volume",
        f"{env['MODEL_STORAGE_PATH']}:/models",
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
    env = _compose_env()
    _preflight(env)
    _ensure_vllm_image(env)
    _ensure_model(env)
    _run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(HERE),
            "--file",
            str(HERE / "compose.yaml"),
            "up",
            "--build",
            "--detach",
            "--wait",
            "--wait-timeout",
            "5400",
        ],
        env=env,
    )
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
