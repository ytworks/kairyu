#!/usr/bin/env python3
"""One-command lifecycle for GLM-5.3-Flash answers decided by Quyet.

GLM-5.3-Flash (one TP2 x DP3 / EP6 replica, GPUs 0-5) answers every request as it
is; two Quyet-1.0-Large replicas (stock vLLM plus this example's System One
adapter) run through System One: quyet-judge (GPU 7) decides whether the answer
would cost the work more replies than needed, and a failing answer is written
once more; quyet-route (GPU 6) stays deployed with the L1 and is not read.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
ROOT = HERE.parents[1]
PROJECT = "kairyu-glm-5-3-flash-quyet-8gpu"
GLM = SPEC["allocation"]["glm"]
GLM_GPU_IDS: list[int] = [int(index) for index in GLM["gpu_ids"]]
# DP rank r owns the TP group at positions [r * TP, (r + 1) * TP) of the
# container's device list (vLLM's DP-major local rank order).
DP_RANK_GPU_IDS: list[list[int]] = [list(group) for group in GLM["dp_rank_gpu_ids"]]
QUYET = SPEC["allocation"]["quyet"]
QUYET_GPU_IDS: list[int] = [int(index) for index in QUYET["gpu_ids"]]
# Two Quyet replicas, each a vLLM on its GPU plus its adapter on the CPU:
# quyet-route (unread since VCO-D27) and quyet-judge (the decision).
QUYET_REPLICAS: list[dict] = list(QUYET["replicas"])
QUYET_SERVICES: list[str] = [replica["service"] for replica in QUYET_REPLICAS]
# The services that hold GPUs.
L1_SERVICES = ("glm", *(replica["vllm_service"] for replica in QUYET_REPLICAS))
PUBLIC_MODELS: list[str] = list(SPEC["public_models"])
# kairyu-verified-tool: the answer, then Quyet's decision.
(MODEL,) = PUBLIC_MODELS
GLM_SERVED = SPEC["glm"]["model"]["served_name"]
GLM_VLLM = SPEC["glm"]["vllm"]
QUYET_SERVED = SPEC["quyet"]["model"]["served_name"]
QUYET_VLLM = SPEC["quyet"]["vllm"]
QUYET_SYSTEMONE = SPEC["quyet"]["systemone"]


def _check_allocation() -> None:
    tp = int(GLM["tensor_parallel_size"])
    dp = int(GLM["data_parallel_size"])
    flat = [index for group in DP_RANK_GPU_IDS for index in group]
    every = GLM_GPU_IDS + QUYET_GPU_IDS
    if (
        len(DP_RANK_GPU_IDS) != dp
        or any(len(group) != tp for group in DP_RANK_GPU_IDS)
        or flat != GLM_GPU_IDS
        or int(GLM["expert_parallel_size"]) != tp * dp
        or len(set(every)) != len(every)
        or [int(replica["gpu_id"]) for replica in QUYET_REPLICAS] != QUYET_GPU_IDS
        or len(every) != int(SPEC["hardware"]["gpu_count"])
    ):
        raise SystemExit(
            "example.json allocation is inconsistent (GLM TP x DP must tile its "
            "GPUs; each Quyet replica takes one distinct GPU)"
        )


_check_allocation()


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


def _nvme_root() -> Path:
    configured = Path(os.environ.get("NVME_STORAGE_ROOT", SPEC["storage"]["root"]))
    if not configured.is_absolute():
        raise SystemExit("NVME_STORAGE_ROOT must be an absolute path below /mnt/nvme")
    root = configured.resolve()
    nvme = Path("/mnt/nvme")
    if root != nvme and nvme not in root.parents:
        raise SystemExit("NVME_STORAGE_ROOT must be /mnt/nvme or one of its descendants")
    return root


def environment_storage() -> Path:
    return _nvme_root() / "model-volumes" / SPEC["environment"]


def _storage_paths() -> dict[str, Path]:
    environment = environment_storage()
    paths = {
        "glm_models": environment / "models" / "glm",
        "quyet_models": environment / "models" / "quyet",
        "glm_cache": environment / "compile-cache" / "glm",
        **{
            f"{replica['service']}_cache": environment / "compile-cache" / replica["service"]
            for replica in QUYET_REPLICAS
        },
        "placement_log": environment / "placement-log",
        "webui": environment / "webui-data",
    }
    for path in paths.values():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(f"cannot prepare NVMe storage {path}: {error}") from error
    return paths


def _quyet_variable(replica: dict, suffix: str) -> str:
    """QUYET_ROUTE_<suffix> / QUYET_JUDGE_<suffix> for a Quyet replica."""

    return f"{replica['service'].upper().replace('-', '_')}_{suffix}"


def quyet_l1_url(env: dict[str, str], service: str) -> str:
    """The replica's System One adapter (host loopback)."""

    replica = next(item for item in QUYET_REPLICAS if item["service"] == service)
    return f"http://127.0.0.1:{env[_quyet_variable(replica, 'L1_PORT')]}"


def quyet_vllm_url(env: dict[str, str], service: str) -> str:
    """The replica's vLLM (host loopback)."""

    replica = next(item for item in QUYET_REPLICAS if item["service"] == service)
    return f"http://127.0.0.1:{env[_quyet_variable(replica, 'VLLM_L1_PORT')]}"


def _compose_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME"}
    }
    paths = _storage_paths()
    env.update({key: str(value) for key, value in QUYET_VLLM["settings"].items()})
    env.update({key: str(value) for key, value in QUYET_SYSTEMONE["settings"].items()})
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": PROJECT,
            "GLM_MODEL_STORAGE_PATH": str(paths["glm_models"]),
            "GLM_CACHE_PATH": str(paths["glm_cache"]),
            "QUYET_MODEL_STORAGE_PATH": str(paths["quyet_models"]),
            "PLACEMENT_LOG_PATH": str(paths["placement_log"]),
            "GLM_VLLM_IMAGE": GLM_VLLM["image"],
            "QUYET_VLLM_IMAGE": QUYET_VLLM["image"],
            "QUYET_SYSTEMONE_IMAGE": QUYET_SYSTEMONE["image"],
            "OPEN_WEBUI_IMAGE": os.environ.get("OPEN_WEBUI_IMAGE", SPEC["webui"]["image"]),
            "WEBUI_STORAGE_PATH": str(paths["webui"]),
            "CHAT_UI_PORT": os.environ.get("CHAT_UI_PORT", str(SPEC["webui"]["port"])),
            "CHAT_UI_BIND_ADDRESS": os.environ.get("CHAT_UI_BIND_ADDRESS", "0.0.0.0"),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "GLM_L1_PORT": os.environ.get("GLM_L1_PORT", str(GLM["l1_port"])),
            **{
                _quyet_variable(replica, suffix): os.environ.get(
                    _quyet_variable(replica, suffix), str(replica[key])
                )
                for replica in QUYET_REPLICAS
                for suffix, key in (
                    ("L1_PORT", "systemone_l1_port"),
                    ("VLLM_L1_PORT", "vllm_l1_port"),
                )
            },
            **{
                _quyet_variable(replica, "CACHE_PATH"): str(paths[f"{replica['service']}_cache"])
                for replica in QUYET_REPLICAS
            },
            # Render-safe defaults for down/status/logs; `up` replaces them
            # with the GPUs' NUMA-local CPU sets.
            "GLM_CPUSET": os.environ.get("GLM_CPUSET", "0"),
            **{
                _quyet_variable(replica, "CPUSET"): os.environ.get(
                    _quyet_variable(replica, "CPUSET"), "0"
                )
                for replica in QUYET_REPLICAS
            },
        }
    )
    return env


def _compose(arguments: list[str], *, env: dict[str, str] | None = None, check: bool = True):
    return _run(
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
        check=check,
    )


def gpu_inventory(text: str) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = [part.strip() for part in raw.split(",")]
        if len(fields) != 6:
            raise SystemExit(f"unexpected nvidia-smi row: {raw}")
        rows[int(fields[0])] = {
            "name": fields[1],
            "memory_mib": int(fields[2]),
            "memory_used_mib": int(fields[3]),
            "compute_capability": float(fields[4]),
            "pci_bus_id": fields[5].lower(),
        }
    return rows


def numa_node(pci_bus_id: str, sysfs: Path = Path("/sys")) -> int:
    canonical = pci_bus_id[4:] if pci_bus_id.startswith("00000000:") else pci_bus_id
    try:
        node = int((sysfs / "bus/pci/devices" / canonical / "numa_node").read_text())
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot determine NUMA node of GPU {pci_bus_id}: {error}") from error
    if node < 0:
        raise SystemExit(f"GPU {pci_bus_id} reports no NUMA node")
    return node


def dp_rank_numa_layout(nodes: dict[int, int], dp_rank_gpu_ids: list[list[int]]) -> list[int]:
    """NUMA node per DP rank; a TP group must sit on one node.

    A TP group split across sockets pays the SYS (inter-socket) PCIe path on
    every all-reduce; the example refuses that layout instead of running slower.
    """

    layout = []
    for rank, group in enumerate(dp_rank_gpu_ids):
        rank_nodes = {nodes[index] for index in group}
        if len(rank_nodes) != 1:
            raise SystemExit(
                f"DP rank {rank} GPUs {group} span NUMA nodes {sorted(rank_nodes)}; "
                "each TP group must share one NUMA node"
            )
        layout.append(rank_nodes.pop())
    return layout


def _node_cpulist(node: int) -> str:
    try:
        cpus = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    except OSError as error:
        raise SystemExit(f"cannot read CPUs of NUMA node {node}: {error}") from error
    if not cpus:
        raise SystemExit(f"NUMA node {node} has no CPUs")
    return cpus


def _held_gpus() -> set[int]:
    """GPUs already assigned to this example's own running L1 containers."""

    held: set[int] = set()
    for service in L1_SERVICES:
        state = _run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}} {{json .HostConfig.DeviceRequests}}",
                f"{PROJECT}-{service}-1",
            ],
            capture=True,
            check=False,
        )
        if state.returncode != 0:
            continue
        running, _, requests = state.stdout.strip().partition(" ")
        if running != "true":
            continue
        for request in json.loads(requests or "null") or ():
            held.update(int(device) for device in request.get("DeviceIDs") or ())
    return held


def _preflight(env: dict[str, str]) -> None:
    for executable in ("docker", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable is missing: {executable}")
    rows = gpu_inventory(
        _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,compute_cap,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture=True,
        ).stdout
    )
    expected = SPEC["hardware"]
    every = GLM_GPU_IDS + QUYET_GPU_IDS
    missing = [index for index in every if index not in rows]
    if missing:
        raise SystemExit(f"GPUs {missing} are not present; this example uses GPUs {every}")
    held = _held_gpus()
    for index in every:
        row = rows[index]
        if row["name"] != expected["product"]:
            raise SystemExit(f"GPU {index} is {row['name']!r}; expected {expected['product']!r}")
        if row["memory_mib"] < expected["minimum_vram_mib"]:
            raise SystemExit(f"GPU {index} has insufficient VRAM")
        if row["compute_capability"] < expected["minimum_compute_capability"]:
            raise SystemExit(f"GPU {index} has insufficient compute capability")
        # Never evict another workload.
        if index not in held and int(row["memory_used_mib"]) > int(
            expected["maximum_idle_used_mib"]
        ):
            raise SystemExit(
                f"GPU {index} already has {row['memory_used_mib']} MiB in use; "
                "stop the workload holding it before starting this example"
            )
    nodes = {index: numa_node(str(rows[index]["pci_bus_id"])) for index in every}
    layout = dp_rank_numa_layout(nodes, DP_RANK_GPU_IDS)
    env["GLM_CPUSET"] = ",".join(_node_cpulist(node) for node in dict.fromkeys(layout))
    placement = {
        "GLM": "; ".join(
            f"DP{rank}=GPU {','.join(map(str, group))} (NUMA {layout[rank]})"
            for rank, group in enumerate(DP_RANK_GPU_IDS)
        )
    }
    for replica in QUYET_REPLICAS:
        node = nodes[int(replica["gpu_id"])]
        env[_quyet_variable(replica, "CPUSET")] = _node_cpulist(node)
        placement[replica["service"]] = f"GPU {replica['gpu_id']} (NUMA {node})"
    print(
        "hardware: " + "; ".join(f"{name} on {where}" for name, where in placement.items()),
        flush=True,
    )


def _image_field(image: str, template: str) -> str | None:
    inspected = _run(
        ["docker", "image", "inspect", "--format", template, image], capture=True, check=False
    )
    return inspected.stdout.strip() if inspected.returncode == 0 else None


def _vllm_image_matches(image: str, repo_digest: str) -> bool:
    """The image is the pinned registry digest. An image ID is not compared: the classic
    store reports the config digest, the containerd store the manifest-list digest."""

    digests = json.loads(_image_field(image, "{{json .RepoDigests}}") or "null") or []
    return repo_digest in digests


def _ensure_vllm_image(image: str, repo_digest: str) -> None:
    """Pull a pinned stock vLLM image when absent, then check its registry digest."""

    if _image_field(image, "{{.Id}}") is None:
        _run(["docker", "pull", image])
    if not _vllm_image_matches(image, repo_digest):
        raise SystemExit(f"local {image} is not {repo_digest}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_labels() -> dict[str, str]:
    """What the Quyet adapter image is built from, recorded as labels on the image."""

    return {
        "org.kairyu.quyet.vllm": QUYET_VLLM["repo_digest"],
        "org.kairyu.quyet.version": QUYET_SYSTEMONE["quyet_version"],
        "org.kairyu.quyet.requirements-sha256": _sha256(HERE / "quyet-requirements.txt"),
        "org.kairyu.quyet.dockerfile-sha256": _sha256(HERE / QUYET_SYSTEMONE["dockerfile"]),
        "org.kairyu.quyet.adapter-sha256": _sha256(HERE / "quyet_systemone.py"),
    }


def _adapter_image_matches(image: str) -> bool:
    labels = json.loads(_image_field(image, "{{json .Config.Labels}}") or "null") or {}
    return all(labels.get(key) == value for key, value in adapter_labels().items())


def _ensure_quyet_images() -> None:
    """Pull the pinned vLLM image; build the adapter on it whenever its sources changed."""

    image = QUYET_VLLM["image"]
    _ensure_vllm_image(image, QUYET_VLLM["repo_digest"])
    adapter = QUYET_SYSTEMONE["image"]
    if _adapter_image_matches(adapter):
        return
    print("building the Quyet System One adapter image from the current sources", flush=True)
    labels = [
        arg for key, value in adapter_labels().items() for arg in ("--label", f"{key}={value}")
    ]
    _run(
        ["docker", "build", "--file", str(HERE / QUYET_SYSTEMONE["dockerfile"]),
         "--build-arg", f"VLLM_IMAGE={image}",
         "--build-arg", f"QUYET_VERSION={QUYET_SYSTEMONE['quyet_version']}",
         *labels, "--tag", adapter, str(HERE)]
    )  # fmt: skip
    if not _adapter_image_matches(adapter):
        raise SystemExit(f"the built {adapter} does not carry the expected source labels")


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

if not (target / 'config.json').is_file():
    snapshot_download(repo, revision=revision, local_dir=target,
                      token=os.environ.get('HF_TOKEN') or None)
rows, tree = inventory()
if tree != expected_tree:
    raise SystemExit(f'checkpoint tree mismatch: {tree}')
attestation.write_text(json.dumps({'schema_version': 1, 'repo': repo,
                                    'revision': revision, 'tree_sha256': tree,
                                    'files': rows}, sort_keys=True))
"""


def _seed_model(target: Path, variable: str, image: str) -> None:
    """Hard-link an already downloaded checkpoint instead of downloading it.

    Checkpoints written by containers are root-owned and the host forbids
    hard links to another user's files, so the link runs in a container with
    /mnt/nvme mounted once (a link cannot cross two bind mounts). The seed is
    trusted for nothing: the copy is re-hashed against this example's pinned
    tree before it is served.
    """

    seed = os.environ.get(variable)
    if not seed or target.exists():
        return
    source = Path(seed).resolve()
    nvme = Path("/mnt/nvme")
    if nvme not in source.parents or not (source / "config.json").is_file():
        raise SystemExit(f"{variable} must name a checkpoint below /mnt/nvme: {source}")
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            "--volume",
            f"{nvme}:{nvme}",
            image,
            "-c",
            'cp -al "$0" "$1" && rm -f "$1/.kairyu-model-attestation.json"',
            str(source),
            str(target),
        ]
    )


def _ensure_model(storage: str, model: dict, image: str) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        "--volume",
        f"{storage}:/models",
        # Hub and Xet caches stay on NVMe, beside the checkpoint.
        "--env",
        "HF_HOME=/models/.cache",
    ]
    if "HF_TOKEN" in os.environ:
        command.extend(["--env", "HF_TOKEN"])
    if os.environ.get("VERIFY_MODEL") == "1":
        command.extend(["--env", "VERIFY_MODEL=1"])
    command.extend(
        [
            image,
            "-c",
            _MODEL_PROGRAM,
            model["repo"],
            model["revision"],
            model["slug"],
            model["tree_sha256"],
        ]
    )
    _run(command)


# Runs inside the adapter image (huggingface_hub is vLLM's). Downloads the pinned
# revision when needed, then checks every file against the publisher's
# MANIFEST.sha256 and the whole tree against example.json.
_QUYET_MODEL_PROGRAM = r"""
import hashlib, json, os, sys
from pathlib import Path

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

def check_manifest(rows):
    listed = {}
    for line in (target / 'MANIFEST.sha256').read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            listed[name.strip()] = digest
    have = {row['path']: row['sha256'] for row in rows}
    wrong = sorted(name for name, digest in listed.items() if have.get(name) != digest)
    if wrong:
        raise SystemExit(f'files differ from the publisher MANIFEST.sha256: {wrong}')

if attestation.exists() and os.environ.get('VERIFY_MODEL') != '1':
    current = json.loads(attestation.read_text())
    pinned = (current.get('repo'), current.get('revision'), current.get('tree_sha256'))
    if pinned != (repo, revision, expected_tree):
        raise SystemExit('model attestation does not match the pinned checkpoint')
    raise SystemExit(0)

if not attestation.exists():
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=revision, local_dir=target,
                      token=os.environ.get('HF_TOKEN') or None)
rows, tree = inventory()
check_manifest(rows)
if tree != expected_tree:
    raise SystemExit(f'checkpoint tree {tree} differs from model.tree_sha256 {expected_tree}')
attestation.write_text(json.dumps({'schema_version': 1, 'repo': repo, 'revision': revision,
                                   'tree_sha256': tree, 'files': rows}, sort_keys=True))
print(f'model attested: {len(rows)} files, tree {tree}')
"""


def required_free_gib(*, glm_present: bool, quyet_present: bool) -> int:
    """NVMe space the checkpoints still to be fetched need: each one's own
    example's minimum (glm-5.3-flash-6gpu, quyet-1.0-large-1gpu), summed."""

    minimum = SPEC["storage"]["minimum_free_gib"]
    return (0 if glm_present else int(minimum["glm"])) + (
        0 if quyet_present else int(minimum["quyet"])
    )


def glm_present(target: Path) -> bool:
    """GLM needs no download: _MODEL_PROGRAM fetches only without config.json.
    A seed counts once _seed_model has linked it, never for its variable alone."""

    return (target / "config.json").is_file()


def quyet_present(target: Path) -> bool:
    """Quyet needs no download: _QUYET_MODEL_PROGRAM fetches only without its attestation."""

    return (target / ".kairyu-model-attestation.json").exists()


def _check_free_space(env: dict[str, str]) -> None:
    """One check before any download (after the seed is linked), so a fetched
    checkpoint never fails the next one's."""

    required = required_free_gib(
        glm_present=glm_present(Path(env["GLM_MODEL_STORAGE_PATH"]) / SPEC["glm"]["model"]["slug"]),
        quyet_present=quyet_present(
            Path(env["QUYET_MODEL_STORAGE_PATH"]) / SPEC["quyet"]["model"]["slug"]
        ),
    )
    free_gib = shutil.disk_usage(environment_storage()).free // (1024**3)
    if free_gib < required:
        raise SystemExit(f"NVMe storage has {free_gib} GiB free; {required} GiB is required")


def _ensure_quyet_model(env: dict[str, str]) -> None:
    """Download (or re-verify) the pinned Quyet checkpoint both replicas read."""

    model = SPEC["quyet"]["model"]
    storage = env["QUYET_MODEL_STORAGE_PATH"]
    command = ["docker", "run", "--rm", "--entrypoint", "python3",
               "--env", "HF_HUB_DISABLE_TELEMETRY=1",
               "--volume", f"{storage}:/models"]  # fmt: skip
    if "HF_TOKEN" in os.environ:
        command.extend(["--env", "HF_TOKEN"])
    if os.environ.get("VERIFY_MODEL") == "1":
        command.extend(["--env", "VERIFY_MODEL=1"])
    command.extend(
        [QUYET_SYSTEMONE["image"], "-c", _QUYET_MODEL_PROGRAM, model["repo"], model["revision"],
         model["slug"], str(model["tree_sha256"])]
    )  # fmt: skip
    _run(command)


def _json_url(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def _text_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def post_json(url: str, payload: dict, *, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def _healthy_replicas(metrics: str, pool: str) -> int | None:
    pattern = re.compile(
        r'^kairyu_pool_replicas\{pool="' + re.escape(pool) + r'",state="healthy"\} (\S+)$',
        re.MULTILINE,
    )
    match = pattern.search(metrics)
    return int(float(match.group(1))) if match else None


def validate_ready(api_url: str) -> None:
    try:
        ready = _json_url(f"{api_url}/readyz")
        listed = _json_url(f"{api_url}/v1/models")
        models = {row["id"] for row in listed["data"]}
        metrics = _text_url(f"{api_url}/metrics")
        healthy = {pool: _healthy_replicas(metrics, pool) for pool in (GLM_SERVED, QUYET_SERVED)}
    except (KeyError, OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Kairyu readiness evidence is incomplete: {error}") from error
    if ready.get("status") != "ready" or models != set(PUBLIC_MODELS):
        raise SystemExit(f"Kairyu must serve exactly {PUBLIC_MODELS!r}, got {sorted(models)!r}")
    expected = {GLM_SERVED: 1, QUYET_SERVED: len(QUYET_REPLICAS)}
    if healthy != expected:
        raise SystemExit(f"healthy replicas must be {expected!r}, got {healthy!r}")


# Readiness checks grammar-constrained JSON (a caller's response_format) at
# the template's default effort (max) and the lightest one on every DP rank.
_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {"answer": {"type": "integer"}},
        },
    },
}


def glm_probe(effort: str | None) -> dict:
    payload: dict = {
        "model": GLM_SERVED,
        "messages": [{"role": "user", "content": 'What is 17 * 19? Answer as JSON {"answer": n}.'}],
        "max_tokens": 16384,
        "response_format": _ANSWER_SCHEMA,
    }
    if effort is not None:
        payload["reasoning_effort"] = effort
    return payload


def glm_answer_error(body: dict) -> str | None:
    try:
        choice = body["choices"][0]
        content = choice["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return f"malformed response: {str(body)[:200]}"
    if choice.get("finish_reason") != "stop":
        return f"finish_reason is {choice.get('finish_reason')!r}"
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return f"content is not JSON: {content!r}"
    if value != {"answer": 323}:
        return f"answer is {value!r}, expected {{'answer': 323}}"
    return None


def _validate_glm(l1_url: str) -> None:
    # One request per DP rank and effort at once so vLLM's DP balancer hands a
    # probe to every rank.
    efforts = [effort for effort in (None, "low") for _ in DP_RANK_GPU_IDS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(efforts)) as pool:
        bodies = list(
            pool.map(
                lambda effort: post_json(
                    f"{l1_url}/v1/chat/completions", glm_probe(effort), timeout_s=900
                ),
                efforts,
            )
        )
    for effort, body in zip(efforts, bodies, strict=True):
        error = glm_answer_error(body)
        if error is not None:
            raise SystemExit(f"GLM JSON probe at effort {effort or 'default'} failed: {error}")


SYSTEMONE_PROBE = {
    "state": "The invoice for March was paid twice by mistake.",
    "questions": {
        "billing": {"type": "noul", "instructions": "This is about a billing problem."},
        "weather": {"type": "noul", "instructions": "This is about the weather."},
    },
}


def systemone_probe_error(body: dict) -> str | None:
    try:
        billing = body["answers"]["billing"]["noul"]
        weather = body["answers"]["weather"]["noul"]
    except (KeyError, TypeError) as error:
        return f"malformed answer ({error!r}): {str(body)[:200]}"
    if not 0.5 < billing <= 1 or not 0 <= weather < 0.5:
        return f"implausible probabilities billing={billing} weather={weather}"
    return None


def _validate_systemone(l1_url: str) -> None:
    """Quyet answers System One on its own adapter (the judge is not a public model)."""

    body = post_json(
        f"{l1_url}/v1/systemone",
        {"model": SPEC["systemone"]["upstream_model"], **SYSTEMONE_PROBE},
        timeout_s=300,
    )
    error = systemone_probe_error(body)
    if error is not None:
        raise SystemExit(f"Quyet System One probe failed: {error}")


# A state well over Quyet's 6,000 tokens: read whole it would be cut.
_LONG_STATE = {"log": "\n".join(f"line {n}: the build step {n} finished" for n in range(3000))}


def _validate_strict(l1_url: str) -> None:
    """The judge replica answers a long state under Quyet's own name (cut, with a
    warning) and refuses it under the decision's strict name."""

    questions = {"done": {"type": "noul", "instructions": "The log reports a finished build."}}
    plain = post_json(
        f"{l1_url}/v1/systemone",
        {"model": QUYET_SYSTEMONE["upstream_model"], "state": _LONG_STATE, "questions": questions},
        timeout_s=300,
    )
    if not any(item.get("code") == "state_truncated" for item in plain.get("warnings") or []):
        raise SystemExit(f"Quyet did not cut the long probe state: {str(plain)[:300]}")
    request = urllib.request.Request(
        f"{l1_url}/v1/systemone",
        data=json.dumps(
            {"model": QUYET_SYSTEMONE["strict_model"], "state": _LONG_STATE, "questions": questions}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read()[:300]
    except urllib.error.HTTPError as error:
        if error.code == 400:
            return
        raise SystemExit(f"strict probe answered {error.code}, expected 400") from error
    raise SystemExit(f"the strict name answered a state Quyet would cut: {body!r}")


def routed_request(content: str, *, model: str = MODEL, **overrides) -> dict:
    """A chat request to the public model unless ``model`` says otherwise."""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        # The roles' own cap (verified-tool.yaml).
        "max_tokens": 262144,
    }
    payload.update(overrides)
    return payload


# A caller tool for the serving probe: the next reply must call it.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def validate_tool_answer(api_url: str) -> None:
    """The public model answers a tool turn with a structured tool call."""

    body = post_json(
        f"{api_url}/v1/chat/completions",
        routed_request("What is the weather in Paris right now?", tools=[WEATHER_TOOL]),
        timeout_s=1800,
    )
    try:
        call = body["choices"][0]["message"]["tool_calls"][0]["function"]
        city = json.loads(call["arguments"]).get("city", "")
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise SystemExit(f"tool-call probe failed: {str(body)[:300]}") from error
    if call.get("name") != "get_weather" or "paris" not in str(city).lower():
        raise SystemExit(f"tool-call probe failed: {call!r}")


_CHAT_UI_EFFORT_FILTER_ID = "reasoning_effort"
_CHAT_UI_EFFORT_LEVELS: list[str] = list(SPEC["webui"]["reasoning_effort_levels"])


def _webui_api(ui_url: str, path: str, *, token: str | None = None, payload: dict | None = None):
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


def provision_chat_ui(ui_url: str) -> None:
    """Install the Reasoning Effort dropdown and check the public model is offered."""

    filter_source = (HERE / "webui-reasoning-effort-filter.py").read_text(encoding="utf-8")
    base = f"/api/v1/functions/id/{_CHAT_UI_EFFORT_FILTER_ID}"
    try:
        signin = _webui_api(ui_url, "/api/v1/auths/signin", payload={"email": "", "password": ""})
        if not isinstance(signin, dict) or signin.get("role") != "admin" or not signin.get("token"):
            raise SystemExit("Chat UI signin did not return the auth-disabled admin session")
        token = signin["token"]
        listed = _webui_api(ui_url, "/api/v1/functions/", token=token)
        existing = next((row for row in listed if row.get("id") == _CHAT_UI_EFFORT_FILTER_ID), None)
        body = {
            "id": _CHAT_UI_EFFORT_FILTER_ID,
            "name": "Reasoning Effort",
            "content": filter_source,
            "meta": {"description": "GLM reasoning effort."},
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
        spec = _webui_api(ui_url, f"{base}/valves/user/spec", token=token)
        enum = spec.get("properties", {}).get("reasoning_effort", {}).get("enum")
        offered = {row["id"] for row in _webui_api(ui_url, "/api/models", token=token)["data"]}
    except (KeyError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Chat UI provisioning failed: {error}") from error
    if not (state.get("is_active") and state.get("is_global")):
        raise SystemExit("Chat UI effort selector could not be activated globally")
    if enum != _CHAT_UI_EFFORT_LEVELS:
        raise SystemExit(f"Chat UI effort selector must expose {_CHAT_UI_EFFORT_LEVELS!r}")
    if not set(PUBLIC_MODELS) <= offered:
        raise SystemExit(f"Chat UI must offer {PUBLIC_MODELS!r}, got {sorted(offered)!r}")


def validate_serving(env: dict[str, str]) -> None:
    api_url = f"http://127.0.0.1:{env['API_PORT']}"
    validate_ready(api_url)
    _validate_glm(f"http://127.0.0.1:{env['GLM_L1_PORT']}")
    for service in QUYET_SERVICES:
        try:
            _json_url(f"{quyet_vllm_url(env, service)}/v1/models")
        except (OSError, ValueError, urllib.error.URLError) as error:
            raise SystemExit(f"{service} vLLM is not serving: {error}") from error
        _validate_systemone(quyet_l1_url(env, service))
    _validate_strict(quyet_l1_url(env, "quyet-judge"))
    validate_tool_answer(api_url)


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


def ui_check_url(env: dict[str, str]) -> str:
    """Where this host reaches the Chat UI: loopback when it listens on every interface."""

    bind = env["CHAT_UI_BIND_ADDRESS"]
    host = "127.0.0.1" if bind == "0.0.0.0" else bind
    return f"http://{host}:{env['CHAT_UI_PORT']}"


def up() -> None:
    env = _compose_env()
    bind = env["CHAT_UI_BIND_ADDRESS"]
    ui_host = _public_ui_host() if bind == "0.0.0.0" else bind
    env.setdefault("WEBUI_URL", f"http://{ui_host}:{env['CHAT_UI_PORT']}")
    _preflight(env)
    _ensure_vllm_image(env["GLM_VLLM_IMAGE"], GLM_VLLM["repo_digest"])
    _ensure_quyet_images()
    # A seed is hard-linked (no space) before the space check, which then sees
    # the checkpoint it left, not the variable.
    _seed_model(
        Path(env["GLM_MODEL_STORAGE_PATH"]) / SPEC["glm"]["model"]["slug"],
        "GLM_MODEL_SEED",
        env["GLM_VLLM_IMAGE"],
    )
    _check_free_space(env)
    _ensure_model(env["GLM_MODEL_STORAGE_PATH"], SPEC["glm"]["model"], env["GLM_VLLM_IMAGE"])
    _ensure_quyet_model(env)
    # --remove-orphans: an update from a release that still defined a service
    # stops and removes its old container.
    _compose(
        ["up", "--build", "--detach", "--wait", "--wait-timeout", "7200", "--remove-orphans"],
        env=env,
    )
    validate_serving(env)
    provision_chat_ui(ui_check_url(env))
    print("\nEnvironment is ready.")
    print(f"Chat UI:     http://{ui_host}:{env['CHAT_UI_PORT']} (Open WebUI, no authentication)")
    print(f"OpenAI API:  http://127.0.0.1:{env['API_PORT']}/v1 (model {MODEL})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("up", "down", "status", "logs"), default="up")
    args = parser.parse_args()
    if args.action == "up":
        up()
    elif args.action == "down":
        _compose(["down", "--remove-orphans"])
    elif args.action == "status":
        _compose(["ps"])
    else:
        _compose(["logs", "--follow", "--tail", "200"])


if __name__ == "__main__":
    main()
