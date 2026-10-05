#!/usr/bin/env python3
"""One-command lifecycle for Winnow-12B Q8_0 (llama.cpp) behind Kairyu."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
ROOT = HERE.parents[1]
REPLICAS = SPEC["replicas"]


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


def _storage_root() -> Path:
    configured = Path(os.environ.get("NVME_STORAGE_ROOT", SPEC["storage"]["root"]))
    if not configured.is_absolute():
        raise SystemExit("NVME_STORAGE_ROOT must be an absolute path below /mnt/nvme")
    root = configured.resolve()
    nvme = Path("/mnt/nvme")
    if root != nvme and nvme not in root.parents:
        raise SystemExit("NVME_STORAGE_ROOT must be /mnt/nvme or one of its descendants")
    return root


def _storage_paths() -> tuple[Path, Path, Path]:
    root = _storage_root()
    # All Winnow examples share one verified download of the pinned GGUF files.
    model = root / "model-volumes" / "winnow-12b-q8" / "models"
    webui = root / "model-volumes" / SPEC["environment"] / "webui-data"
    placement = root / "model-volumes" / SPEC["environment"] / "placement-log"
    for path in (model, webui, placement):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(f"cannot prepare NVMe storage {path}: {error}") from error
    return model, webui, placement


def _require_free_space() -> None:
    # Only `up` writes (image build, model download); `down`, `status` and
    # `logs` must keep working on a nearly full disk.
    free_gib = shutil.disk_usage(_storage_root()).free // (1024**3)
    minimum = int(SPEC["storage"]["minimum_free_gib"])
    if free_gib < minimum:
        raise SystemExit(f"NVMe storage has {free_gib} GiB free; {minimum} GiB is required")


def placement_log() -> Path:
    """Host path of the pool's placement JSONL (bind-mounted in compose.yaml)."""

    return _storage_paths()[2] / f"{SPEC['model']['served_name']}.jsonl"


def _compose_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME"}
    }
    model_path, webui_path, placement_path = _storage_paths()
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": SPEC["environment"].replace(".", "-"),
            "MODEL_STORAGE_PATH": str(model_path),
            "WEBUI_STORAGE_PATH": str(webui_path),
            "PLACEMENT_LOG_PATH": str(placement_path),
            "WINNOW_IMAGE": os.environ.get("WINNOW_IMAGE", SPEC["runtime"]["image"]),
            "OPEN_WEBUI_IMAGE": os.environ.get("OPEN_WEBUI_IMAGE", SPEC["webui"]["image"]),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "CHAT_UI_PORT": os.environ.get("CHAT_UI_PORT", str(SPEC["webui"]["port"])),
            "CHAT_UI_BIND_ADDRESS": os.environ.get("CHAT_UI_BIND_ADDRESS", "0.0.0.0"),
            "PLAYGROUND_IMAGE": os.environ.get("PLAYGROUND_IMAGE", SPEC["playground"]["image"]),
            "PLAYGROUND_PORT": os.environ.get("PLAYGROUND_PORT", str(SPEC["playground"]["port"])),
        }
    )
    if len(REPLICAS) == 1:
        env["GPU_ID"] = os.environ.get("GPU_ID", str(REPLICAS[0]["gpu"]))
    # `up` replaces these with each GPU's NUMA-local CPUs. A valid default
    # lets `down/status/logs` render Compose without the hardware preflight.
    for index, _replica in enumerate(REPLICAS):
        env.setdefault(f"GPU_CPUSET_{index}", "0")
    return env


def _compose(arguments: list[str], *, env: dict[str, str] | None = None) -> None:
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
        env=env or _compose_env(),
    )


def _gpu_inventory(text: str) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = [part.strip() for part in raw.split(",")]
        if len(fields) != 5:
            raise SystemExit(f"unexpected nvidia-smi row: {raw}")
        rows[int(fields[0])] = {
            "name": fields[1],
            "memory_mib": int(fields[2]),
            "compute_capability": float(fields[3]),
            "pci_bus_id": fields[4].lower(),
        }
    return rows


def _numa_cpuset(pci_bus_id: str) -> str:
    canonical = pci_bus_id[4:] if pci_bus_id.startswith("00000000:") else pci_bus_id
    try:
        node = int((Path("/sys/bus/pci/devices") / canonical / "numa_node").read_text())
        if node < 0:
            raise ValueError("negative NUMA node")
        cpuset = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot determine NUMA affinity for GPU {pci_bus_id}: {error}") from error
    if not cpuset:
        raise SystemExit(f"NUMA CPU set for GPU {pci_bus_id} is empty")
    return cpuset


def _preflight(env: dict[str, str]) -> None:
    for executable in ("docker", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable is missing: {executable}")
    rows = _gpu_inventory(
        _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture=True,
        ).stdout
    )
    expected = SPEC["hardware"]
    for index, replica in enumerate(REPLICAS):
        gpu = int(env["GPU_ID"]) if len(REPLICAS) == 1 else int(replica["gpu"])
        row = rows.get(gpu)
        if row is None:
            raise SystemExit(f"GPU {gpu} is unavailable")
        if row["name"] != expected["product"]:
            raise SystemExit(f"GPU {gpu} is {row['name']!r}; expected {expected['product']!r}")
        if row["memory_mib"] < expected["minimum_vram_mib"]:
            raise SystemExit(f"GPU {gpu} has insufficient VRAM")
        if row["compute_capability"] < expected["minimum_compute_capability"]:
            raise SystemExit(f"GPU {gpu} has insufficient compute capability")
        env[f"GPU_CPUSET_{index}"] = _numa_cpuset(str(row["pci_bus_id"]))
        print(f"hardware: {replica['service']} on GPU {gpu}, cpuset {env[f'GPU_CPUSET_{index}']}")


def _ensure_winnow_image(env: dict[str, str]) -> None:
    image = env["WINNOW_IMAGE"]
    if _run(["docker", "image", "inspect", image], capture=True, check=False).returncode == 0:
        return
    runtime = SPEC["runtime"]
    if image != runtime["image"]:
        raise SystemExit(f"WINNOW_IMAGE does not exist locally: {image}")
    print("winnow-server image is absent; building the pinned source revision", flush=True)
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "winnow-inference"
        _run(["git", "clone", "--quiet", runtime["source_repository"], str(source)])
        _run(
            [
                "git",
                "-C",
                str(source),
                "checkout",
                "--quiet",
                "--detach",
                runtime["source_revision"],
            ]
        )
        _add_runtime_patches(source)
        _run(
            [
                "docker",
                "build",
                "--build-arg",
                f"CUDA_ARCH={runtime['cuda_arch']}",
                "--tag",
                image,
                "--label",
                f"org.opencontainers.image.revision={runtime['source_revision']}",
                "--label",
                "io.kairyu.winnow.extra-patches="
                + ",".join(item["upstream_commit"] for item in runtime["extra_patches"]),
                str(source),
            ]
        )


def _add_runtime_patches(source: Path) -> None:
    """Add upstream llama.cpp backports to Winnow's own patch lock.

    Winnow's build applies every patch listed in ``runtime.lock.json`` and
    rejects any llama.cpp file change missing from its ``source_sha256``.
    ``f072b10`` (after b11036) makes Gemma 4 ``tool_choice: "required"``
    force a tool call; without it llama-server ignores ``required``, which
    the llamacpp profile uses for named tool choices.
    """

    lock_path = source / "runtime.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for item in SPEC["runtime"]["extra_patches"]:
        patch = (HERE / "winnow-patches" / item["file"]).read_bytes()
        if hashlib.sha256(patch).hexdigest() != item["sha256"]:
            raise SystemExit(f"{item['file']} does not match example.json")
        (source / "patches" / item["file"]).write_bytes(patch)
        lock["patches"].append({"file": f"patches/{item['file']}", "sha256": item["sha256"]})
        overlap = set(lock["source_sha256"]) & set(item["source_sha256"])
        if overlap:
            raise SystemExit(f"{item['file']} overlaps Winnow's patches: {sorted(overlap)}")
        lock["source_sha256"].update(item["source_sha256"])
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def _ensure_model(env: dict[str, str]) -> None:
    """Download (or re-verify) the pinned GGUF files with Winnow's own tool.

    ``scripts/download.py`` fetches the manifest's revision-pinned URLs,
    resumes partial transfers and checks every file's size and SHA-256.
    The example's own pins must match that manifest.
    """

    owner = f"{os.getuid()}:{os.getgid()}"
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            owner,
            "--volume",
            f"{env['MODEL_STORAGE_PATH']}:/models",
            "--entrypoint",
            "python3",
            env["WINNOW_IMAGE"],
            "scripts/download.py",
            "--model-dir",
            "/models",
        ]
    )
    manifest = json.loads(
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "cat",
                env["WINNOW_IMAGE"],
                "manifests/models.json",
            ],
            capture=True,
        ).stdout
    )
    release = manifest["release"]
    pinned = SPEC["model"]
    shipped = {
        release[kind]["file"]: {
            "bytes": release[kind]["bytes"],
            "sha256": release[kind]["sha256"],
        }
        for kind in ("model", "projector")
    }
    if manifest.get("model_revision") != pinned["revision"] or shipped != pinned["files"]:
        raise SystemExit("winnow-inference manifest differs from example.json model pins")


def _ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


_NO_PUBLIC_HOST = "cannot discover an externally reachable UI host; set PUBLIC_HOST"


def _public_ui_host() -> str:
    configured = os.environ.get("PUBLIC_HOST", "").strip()
    if configured:
        if "://" in configured or "/" in configured:
            raise SystemExit("PUBLIC_HOST must be a hostname or IPv4 address, not a URL")
        return configured
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            # UDP connect selects the outward-facing interface without sending data.
            route.connect(("192.0.2.1", 9))
            detected = str(route.getsockname()[0])
    except OSError as error:
        raise SystemExit(_NO_PUBLIC_HOST) from error
    if detected == "0.0.0.0" or detected.startswith("127."):
        raise SystemExit(_NO_PUBLIC_HOST)
    return detected


def up() -> None:
    env = _compose_env()
    _require_free_space()
    bind = env["CHAT_UI_BIND_ADDRESS"]
    ui_host = _public_ui_host() if bind == "0.0.0.0" else bind
    env.setdefault("WEBUI_URL", f"http://{ui_host}:{env['CHAT_UI_PORT']}")
    _preflight(env)
    _ensure_winnow_image(env)
    _ensure_model(env)
    _compose(["up", "--build", "--detach", "--wait", "--wait-timeout", "3600"], env=env)
    api_url = f"http://127.0.0.1:{env['API_PORT']}"
    if not _ready(f"{api_url}/readyz"):
        raise SystemExit("Kairyu did not become ready")
    print("\nEnvironment is ready.")
    print(f"OpenAI API: {api_url}/v1  (model {SPEC['model']['served_name']})")
    print(f"System One: {api_url}/v1/systemone  (model winnow-12b-systemone)")
    print(f"Chat UI:    http://{ui_host}:{env['CHAT_UI_PORT']}")
    print(f"Playground: http://{ui_host}:{env['PLAYGROUND_PORT']}  (System One, no authentication)")


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
