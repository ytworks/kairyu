#!/usr/bin/env python3
"""Fail-closed lifecycle controller shared by all frontier examples."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def _run(command: list[str], *, env: dict[str, str] | None = None, capture: bool = False):
    return subprocess.run(
        command,
        check=True,
        text=True,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _load_env(path: Path) -> dict[str, str]:
    values = dict(os.environ)
    if not path.exists():
        return values
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"{path}:{number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        values.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return values


def _save_runtime_env(path: Path, env: dict[str, str]) -> None:
    keys = ("KAIRYU_CPU_IMAGE", "KAIRYU_GPU_IMAGE", "MODEL_VOLUME")
    path.write_text(
        "".join(f"{key}={env[key]}\n" for key in keys if env.get(key)),
        encoding="utf-8",
    )


def _model_storage_directory(spec: dict, env: dict[str, str], root: Path) -> Path:
    configured = env.get("MODEL_STORAGE_ROOT", "").strip()
    if not configured:
        return root
    storage_root = Path(configured).expanduser()
    if not storage_root.is_absolute():
        raise SystemExit("MODEL_STORAGE_ROOT must be an absolute path")
    directory = storage_root / spec["environment"]
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SystemExit(f"cannot create model storage directory {directory}: {error}") from error
    if not directory.is_dir():
        raise SystemExit(f"model storage path is not a directory: {directory}")
    return directory.resolve()


def _gpu_inventory() -> dict[int, dict[str, object]]:
    query = "index,memory.total,compute_cap,pci.bus_id"
    try:
        result = _run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"NVIDIA preflight failed: {error}") from error
    inventory: dict[int, dict[str, object]] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise SystemExit(f"unexpected nvidia-smi row: {line!r}")
        index, memory, capability, bus = fields
        inventory[int(index)] = {
            "memory_mib": int(memory),
            "compute_capability": float(capability),
            "pci_bus_id": bus.lower(),
        }
    return inventory


def _numa_node(pci_bus_id: str) -> int:
    canonical = pci_bus_id
    if canonical.startswith("00000000:"):
        canonical = canonical[4:]
    path = Path("/sys/bus/pci/devices") / canonical / "numa_node"
    try:
        node = int(path.read_text().strip())
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot resolve NUMA node for GPU at {pci_bus_id}: {error}") from error
    if node < 0:
        raise SystemExit(f"GPU at {pci_bus_id} has no NUMA node; affinity cannot be pinned")
    return node


def _preflight(spec: dict, env: dict[str, str], root: Path) -> dict[str, str]:
    if shutil.which("docker") is None:
        raise SystemExit("docker is required")
    _run(["docker", "compose", "version"], capture=True)
    inventory = _gpu_inventory()
    selected = spec["gpu_indices"]
    if spec.get("selectable_gpu"):
        selected = [int(env.get("GPU_ID", str(selected[0])))]
    missing = sorted(set(selected) - set(inventory))
    if missing:
        raise SystemExit(f"required GPU indices are unavailable: {missing}")
    for index in selected:
        gpu = inventory[index]
        if gpu["compute_capability"] < spec["minimum_compute_capability"]:
            raise SystemExit(
                f"GPU {index} compute capability {gpu['compute_capability']} is below "
                f"SM{int(spec['minimum_compute_capability'] * 10)}"
            )
        if gpu["memory_mib"] < spec["minimum_vram_mib"]:
            raise SystemExit(
                f"GPU {index} has {gpu['memory_mib']} MiB; "
                f"{spec['minimum_vram_mib']} MiB is required (no context shrink is allowed)"
            )
    selected_cpus: list[str] = []
    for index in selected:
        node = _numa_node(str(inventory[index]["pci_bus_id"]))
        cpu_list = Path(f"/sys/devices/system/node/node{node}/cpulist")
        cpuset = cpu_list.read_text().strip()
        env[f"GPU_{index}_CPUSET"] = cpuset
        env[f"GPU_{index}_NUMA"] = str(node)
        if cpuset not in selected_cpus:
            selected_cpus.append(cpuset)
    env["SELECTED_GPU_CPUSET"] = ",".join(selected_cpus)
    if len(selected) == 8:
        for left, right in zip(selected[::2], selected[1::2], strict=True):
            left_node = _numa_node(str(inventory[left]["pci_bus_id"]))
            right_node = _numa_node(str(inventory[right]["pci_bus_id"]))
            if left_node != right_node:
                raise SystemExit(f"GPU pair {left}-{right} crosses NUMA nodes")
            cpu_list = Path(f"/sys/devices/system/node/node{left_node}/cpulist")
            env[f"GPU_PAIR_{left}_{right}_CPUSET"] = cpu_list.read_text().strip()
            env[f"GPU_PAIR_{left}_{right}_NUMA"] = str(left_node)
    storage_directory = _model_storage_directory(spec, env, root)
    free_gib = shutil.disk_usage(storage_directory).free // (1024**3)
    if free_gib < spec["minimum_free_disk_gib"]:
        raise SystemExit(
            f"only {free_gib} GiB free at {storage_directory}; "
            f"{spec['minimum_free_disk_gib']} GiB is required"
        )
    for image in spec["images"].values():
        if "@sha256:" not in image:
            raise SystemExit(f"floating container image is forbidden: {image}")
    env["SELECTED_GPU_IDS"] = ",".join(str(index) for index in selected)
    return env


def _build_kairyu_images(
    root: Path,
    spec: dict,
    env: dict[str, str],
    *,
    flavors: tuple[str, ...],
) -> None:
    vcs_ref = _run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    allowed = set(spec.get("kairyu_images", ()))
    for flavor in flavors:
        if flavor not in allowed:
            raise SystemExit(f"example does not declare a Kairyu {flavor} image")
        dockerfile = "Dockerfile.cuda" if flavor == "gpu" else "Dockerfile"
        tag = f"kairyu-example-{flavor}:{vcs_ref[:12]}"
        _run(
            [
                "docker",
                "build",
                "--file",
                str(root / dockerfile),
                "--build-arg",
                f"KAIRYU_VCS_REF={vcs_ref}",
                "--tag",
                tag,
                str(root),
            ],
            env=env,
        )
        image_id = _run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
            capture=True,
        ).stdout.strip()
        if not image_id.startswith("sha256:"):
            raise SystemExit(f"could not lock built Kairyu {flavor} image")
        env[f"KAIRYU_{flavor.upper()}_IMAGE"] = image_id


_DOWNLOAD_PROGRAM = r"""
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
repo, revision, slug = sys.argv[1:]
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
    if current.get('repo') == repo and current.get('revision') == revision:
        snapshot_download(repo, revision=revision, local_dir=target, local_files_only=True)
        rows, tree = inventory()
        if current.get('tree_sha256') != tree or current.get('files') != rows:
            raise SystemExit('local model volume differs from its pinned attestation')
        raise SystemExit(0)
snapshot_download(repo, revision=revision, local_dir=target,
                  token=os.environ.get('HF_TOKEN') or None)
rows, tree = inventory()
attestation.write_text(json.dumps({'schema_version': 1, 'repo': repo, 'revision': revision,
                                    'tree_sha256': tree, 'files': rows}, sort_keys=True))
"""


def _ensure_model_volume(spec: dict, env: dict[str, str], root: Path) -> str:
    volume = f"kairyu-models-{spec['environment']}"
    configured = env.get("MODEL_STORAGE_ROOT", "").strip()
    if not configured:
        _run(["docker", "volume", "create", volume], capture=True)
        return volume
    directory = _model_storage_directory(spec, env, root)
    options = {
        "device": str(directory),
        "o": "bind",
        "type": "none",
    }
    command = ["docker", "volume", "create", "--driver", "local"]
    for key in ("type", "o", "device"):
        command.extend(["--opt", f"{key}={options[key]}"])
    command.append(volume)
    _run(command, capture=True)
    inspection = _run(["docker", "volume", "inspect", volume], capture=True)
    payload = json.loads(inspection.stdout)
    if len(payload) != 1 or payload[0].get("Options") != options:
        raise SystemExit(
            f"existing model volume {volume} is not bound to {directory}; "
            "remove or migrate it before setting MODEL_STORAGE_ROOT"
        )
    return volume


def _download_command(
    volume: str,
    image: str,
    model: dict,
    env: dict[str, str],
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
    ]
    if env.get("HF_TOKEN"):
        # Forward by name so the credential is never embedded in argv, reports,
        # or a CalledProcessError rendered by the lifecycle controller.
        command.extend(["--env", "HF_TOKEN"])
    command.extend(
        [
            "--env",
            "HF_HUB_DISABLE_TELEMETRY=1",
            "--volume",
            f"{volume}:/models",
            image,
            "-c",
            _DOWNLOAD_PROGRAM,
            model["repo"],
            model["revision"],
            model["slug"],
        ]
    )
    return command


def _materialize_models(spec: dict, env: dict[str, str], backend: str, root: Path) -> None:
    volume = _ensure_model_volume(spec, env, root)
    env["MODEL_VOLUME"] = volume
    image = (
        env["KAIRYU_GPU_IMAGE"]
        if backend == "kairyu"
        else env[spec["vllm_download_image_env"]]
    )
    for model in spec["models"]:
        command = _download_command(volume, image, model, env)
        _run(command, env=env)


def _compose(spec_dir: Path, spec: dict, env: dict[str, str], backend: str, args: list[str]):
    compose_env = dict(env)
    profile = backend
    if backend == "kairyu" and spec["environment"] == "deepseek-v4-flash-0731-8gpu":
        topology = env.get("KAIRYU_TOPOLOGY", "ep4")
        if topology not in {"ep4", "ep8"}:
            raise SystemExit("KAIRYU_TOPOLOGY must be ep4 or ep8")
        if topology == "ep8":
            profile = "kairyu-ep8"
    compose_env["COMPOSE_PROJECT_NAME"] = (
        f"kairyu-{spec['environment']}-{profile}"
    )
    compose_env["COMPOSE_PROFILES"] = profile
    compose_env["BACKEND"] = backend
    command = ["docker", "compose"]
    if (spec_dir / ".env").exists():
        command.extend(["--env-file", str(spec_dir / ".env")])
    command.extend(args)
    return _run(command, env=compose_env)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def _wait_ready(spec: dict, backend: str, env: dict[str, str]) -> None:
    port = int(env.get("PORT", spec["port"]))
    root = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + int(env.get("READINESS_TIMEOUT_S", "3600"))
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            ready = _get_json(root + "/readyz")
            models = _get_json(root + "/v1/models")
            backends = _get_json(root + "/backends")
            if ready.get("status") != "ready" or not models.get("data"):
                raise RuntimeError("readiness or model inventory is incomplete")
            serialized = json.dumps(backends, sort_keys=True)
            for model in spec["models"]:
                if model["revision"] not in serialized:
                    raise RuntimeError(f"/backends omits pinned revision for {model['repo']}")
                if str(model["max_context_tokens"]) not in serialized:
                    raise RuntimeError(f"/backends omits native context for {model['repo']}")
            print(f"ready: {root}/v1 ({backend})")
            return
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(2)
    raise SystemExit(f"readiness deadline expired: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("environment_dir", type=Path)
    parser.add_argument("backend", choices=("vllm", "kairyu"))
    parser.add_argument("action", nargs="?", default="up", choices=("up", "down", "status", "logs"))
    args = parser.parse_args()
    spec_dir = args.environment_dir.resolve()
    root = spec_dir.parents[1]
    spec = json.loads((spec_dir / "example.json").read_text(encoding="utf-8"))
    env = _load_env(spec_dir / ".env")
    runtime_env = _load_env(spec_dir / ".runtime.env")
    for key in ("KAIRYU_CPU_IMAGE", "KAIRYU_GPU_IMAGE", "MODEL_VOLUME"):
        if key in runtime_env:
            env[key] = runtime_env[key]
    env.update({key: str(value) for key, value in spec["images"].items()})
    env["PORT"] = env.get("PORT", str(spec["port"]))
    env.setdefault("MODEL_VOLUME", f"kairyu-models-{spec['environment']}")
    if args.action == "down":
        if "KAIRYU_CPU_IMAGE" not in env:
            raise SystemExit("environment has not been built; no stack to stop")
        env.setdefault("KAIRYU_GPU_IMAGE", env["KAIRYU_CPU_IMAGE"])
        _compose(spec_dir, spec, env, args.backend, ["down", "--remove-orphans"])
        return
    if args.action == "status":
        if "KAIRYU_CPU_IMAGE" not in env:
            raise SystemExit("environment has not been built")
        env.setdefault("KAIRYU_GPU_IMAGE", env["KAIRYU_CPU_IMAGE"])
        _compose(spec_dir, spec, env, args.backend, ["ps"])
        return
    if args.action == "logs":
        if "KAIRYU_CPU_IMAGE" not in env:
            raise SystemExit("environment has not been built")
        env.setdefault("KAIRYU_GPU_IMAGE", env["KAIRYU_CPU_IMAGE"])
        _compose(spec_dir, spec, env, args.backend, ["logs", "--follow"])
        return
    env = _preflight(spec, env, root)
    flavors = ("cpu", "gpu") if args.backend == "kairyu" else ("cpu",)
    _build_kairyu_images(root, spec, env, flavors=flavors)
    env.setdefault("KAIRYU_GPU_IMAGE", env["KAIRYU_CPU_IMAGE"])
    _materialize_models(spec, env, args.backend, root)
    _save_runtime_env(spec_dir / ".runtime.env", env)
    _compose(spec_dir, spec, env, args.backend, ["up", "--detach", "--pull", "missing"])
    _wait_ready(spec, args.backend, env)


if __name__ == "__main__":
    main()
