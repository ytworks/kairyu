#!/usr/bin/env python3
"""One-command lifecycle for routed DeepSeek-V4.1 answers.

DeepSeek-V4.1-Flash (one DP6 / EP6 replica, GPUs 0-5) writes; Qwen3.8-27B
(GPU 6) is served as an internal pool; Winnow-12B (GPU 7, llama.cpp) routes
each request through System One to the verified route (max effort) or to one
answer at the caller's effort.
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
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
ROOT = HERE.parents[1]
PROJECT = "kairyu-deepseek-v4-1-qwen3-8-winnow-8gpu"
DEEPSEEK = SPEC["allocation"]["deepseek"]
DEEPSEEK_GPU_IDS: list[int] = [int(index) for index in DEEPSEEK["gpu_ids"]]
DP_RANK_GPU_IDS: list[list[int]] = [list(group) for group in DEEPSEEK["dp_rank_gpu_ids"]]
QWEN_GPU_IDS: list[int] = [int(index) for index in SPEC["allocation"]["qwen"]["gpu_ids"]]
WINNOW_GPU_IDS: list[int] = [int(index) for index in SPEC["allocation"]["winnow"]["gpu_ids"]]
L1_SERVICES = ("deepseek", "qwen", "winnow")
PUBLIC_MODELS: list[str] = list(SPEC["public_models"])
# kairyu-verified: Winnow routes per request; kairyu-verified-always: always
# the verified route.
ROUTED_MODEL, ALWAYS_MODEL = PUBLIC_MODELS
DEEPSEEK_SERVED = SPEC["deepseek"]["served_name"]
QWEN_SERVED = SPEC["qwen"]["served_name"]
WINNOW_SERVED = SPEC["winnow"]["model"]["served_name"]


def _check_allocation() -> None:
    tp = int(DEEPSEEK["tensor_parallel_size"])
    dp = int(DEEPSEEK["data_parallel_size"])
    flat = [index for group in DP_RANK_GPU_IDS for index in group]
    every = DEEPSEEK_GPU_IDS + QWEN_GPU_IDS + WINNOW_GPU_IDS
    if (
        len(DP_RANK_GPU_IDS) != dp
        or any(len(group) != tp for group in DP_RANK_GPU_IDS)
        or flat != DEEPSEEK_GPU_IDS
        or int(DEEPSEEK["expert_parallel_size"]) != tp * dp
        or len(set(every)) != len(every)
        or len(QWEN_GPU_IDS) != 1
        or len(WINNOW_GPU_IDS) != 1
        or len(every) != int(SPEC["hardware"]["gpu_count"])
    ):
        raise SystemExit(
            "example.json allocation is inconsistent (DeepSeek TP x DP must tile its "
            "GPUs; Qwen and Winnow take one distinct GPU each)"
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
        "deepseek_models": environment / "models" / "deepseek",
        "qwen_models": environment / "models" / "qwen",
        "winnow_models": environment / "models" / "winnow",
        "deepseek_cache": environment / "compile-cache" / "deepseek",
        "qwen_cache": environment / "compile-cache" / "qwen",
        "placement_log": environment / "placement-log",
        "webui": environment / "webui-data",
    }
    for path in paths.values():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(f"cannot prepare NVMe storage {path}: {error}") from error
    return paths


def _compose_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME"}
    }
    paths = _storage_paths()
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": PROJECT,
            "DEEPSEEK_MODEL_STORAGE_PATH": str(paths["deepseek_models"]),
            "DEEPSEEK_CACHE_PATH": str(paths["deepseek_cache"]),
            "QWEN_MODEL_STORAGE_PATH": str(paths["qwen_models"]),
            "QWEN_CACHE_PATH": str(paths["qwen_cache"]),
            "WINNOW_MODEL_STORAGE_PATH": str(paths["winnow_models"]),
            "PLACEMENT_LOG_PATH": str(paths["placement_log"]),
            "DEEPSEEK_VLLM_IMAGE": os.environ.get("DEEPSEEK_VLLM_IMAGE", SPEC["deepseek"]["image"]),
            "QWEN_VLLM_IMAGE": os.environ.get("QWEN_VLLM_IMAGE", SPEC["qwen"]["image"]),
            "WINNOW_IMAGE": os.environ.get("WINNOW_IMAGE", SPEC["winnow"]["runtime"]["image"]),
            "PLAYGROUND_IMAGE": os.environ.get("PLAYGROUND_IMAGE", SPEC["playground"]["image"]),
            "OPEN_WEBUI_IMAGE": os.environ.get("OPEN_WEBUI_IMAGE", SPEC["webui"]["image"]),
            "WEBUI_STORAGE_PATH": str(paths["webui"]),
            "CHAT_UI_PORT": os.environ.get("CHAT_UI_PORT", str(SPEC["webui"]["port"])),
            "CHAT_UI_BIND_ADDRESS": os.environ.get("CHAT_UI_BIND_ADDRESS", "0.0.0.0"),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "PLAYGROUND_PORT": os.environ.get("PLAYGROUND_PORT", str(SPEC["playground"]["port"])),
            "PLAYGROUND_BIND_ADDRESS": os.environ.get("PLAYGROUND_BIND_ADDRESS", "0.0.0.0"),
            "DEEPSEEK_L1_PORT": os.environ.get("DEEPSEEK_L1_PORT", "8014"),
            "QWEN_L1_PORT": os.environ.get("QWEN_L1_PORT", "8015"),
            "WINNOW_L1_PORT": os.environ.get("WINNOW_L1_PORT", "8016"),
            # Render-safe defaults for down/status/logs; `up` replaces them
            # with the GPUs' NUMA-local CPU sets.
            "DEEPSEEK_CPUSET": os.environ.get("DEEPSEEK_CPUSET", "0"),
            "QWEN_CPUSET": os.environ.get("QWEN_CPUSET", "0"),
            "WINNOW_CPUSET": os.environ.get("WINNOW_CPUSET", "0"),
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


def _node_cpulist(node: int) -> str:
    try:
        cpus = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    except OSError as error:
        raise SystemExit(f"cannot read CPUs of NUMA node {node}: {error}") from error
    if not cpus:
        raise SystemExit(f"NUMA node {node} has no CPUs")
    return cpus


def _host_memory_available_gib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // (1024 * 1024)
    raise SystemExit("cannot read MemAvailable from /proc/meminfo")


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
    every = DEEPSEEK_GPU_IDS + QWEN_GPU_IDS + WINNOW_GPU_IDS
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
    if not set(DEEPSEEK_GPU_IDS) <= held:
        available = _host_memory_available_gib()
        required = int(expected["minimum_host_available_gib"])
        if available < required:
            raise SystemExit(
                f"host has {available} GiB available; the pinned Engram tables need {required} GiB"
            )
    placement = {}
    for name, gpus in (
        ("DEEPSEEK", DEEPSEEK_GPU_IDS),
        ("QWEN", QWEN_GPU_IDS),
        ("WINNOW", WINNOW_GPU_IDS),
    ):
        nodes = [numa_node(str(rows[index]["pci_bus_id"])) for index in gpus]
        env[f"{name}_CPUSET"] = ",".join(_node_cpulist(node) for node in dict.fromkeys(nodes))
        placement[name.capitalize()] = f"GPUs {gpus} (NUMA {sorted(set(nodes))})"
    print(
        "hardware: " + "; ".join(f"{name} on {where}" for name, where in placement.items()),
        flush=True,
    )


def _image_id(image: str) -> str | None:
    inspected = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture=True,
        check=False,
    )
    return inspected.stdout.strip() if inspected.returncode == 0 else None


def _ensure_deepseek_image(env: dict[str, str]) -> None:
    """Build this example's SM120 overlay when absent, then attest it by ID."""

    image = env["DEEPSEEK_VLLM_IMAGE"]
    source = SPEC["deepseek"]
    actual = _image_id(image)
    if actual is None:
        if image != source["image"]:
            raise SystemExit(f"DEEPSEEK_VLLM_IMAGE does not exist locally: {image}")
        print("vLLM image is absent; building this example's SM120 overlay", flush=True)
        _run(
            [
                "docker",
                "build",
                "--pull",
                "--file",
                str(HERE / source["dockerfile"]),
                "--build-arg",
                f"VLLM_BASE_IMAGE={source['base_image']}",
                "--tag",
                image,
                "--label",
                f"org.opencontainers.image.revision={source['source_revision']}",
                str(HERE),
            ]
        )
        actual = _image_id(image)
    if os.environ.get("DEEPSEEK_ALLOW_UNPINNED_IMAGE") == "1":
        print(f"WARNING: serving unpinned image {image} ({actual}); not evidence", flush=True)
        return
    if actual != source["image_id"]:
        raise SystemExit(
            f"vLLM image {image} has ID {actual}; example.json and kairyu.yaml pin "
            f"{source['image_id']} (update both to this build before serving)"
        )


def _ensure_qwen_image(env: dict[str, str]) -> None:
    """Pull the pinned upstream vLLM release by digest and attest it.

    The pin is the registry digest. The containerd image store reports it as
    the image ID; the classic store reports the config digest as the ID and
    keeps the registry digest in RepoDigests, so either match attests it.
    """

    image = env["QWEN_VLLM_IMAGE"]
    if _image_id(image) is None:
        if image != SPEC["qwen"]["image"]:
            raise SystemExit(f"QWEN_VLLM_IMAGE does not exist locally: {image}")
        _run(["docker", "pull", image])
    inspected = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}} {{json .RepoDigests}}", image],
        capture=True,
    ).stdout.strip()
    actual, _, digests = inspected.partition(" ")
    pinned = SPEC["qwen"]["image_id"]
    if actual != pinned and not any(
        digest.endswith(f"@{pinned}") for digest in json.loads(digests or "null") or ()
    ):
        raise SystemExit(f"Qwen vLLM image {image} ({actual}) is not the pinned image {pinned}")


def _ensure_winnow_image(env: dict[str, str]) -> None:
    """Build winnow-server at the pinned revision with the example's backport."""

    image = env["WINNOW_IMAGE"]
    if _image_id(image) is not None:
        return
    runtime = SPEC["winnow"]["runtime"]
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
        _add_winnow_patches(source)
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


def _add_winnow_patches(source: Path) -> None:
    """Add upstream llama.cpp backports to Winnow's own patch lock.

    Winnow's build applies every patch listed in ``runtime.lock.json`` and
    rejects any llama.cpp file change missing from its ``source_sha256``.
    """

    lock_path = source / "runtime.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for item in SPEC["winnow"]["runtime"]["extra_patches"]:
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


def _ensure_model(storage: str, model: dict, image: str, seed_variable: str) -> None:
    target = Path(storage) / model["slug"]
    _seed_model(target, seed_variable, image)
    if not (target / "config.json").is_file():
        free_gib = shutil.disk_usage(target.parent).free // (1024**3)
        minimum = int(SPEC["storage"]["minimum_free_gib"])
        if free_gib < minimum:
            raise SystemExit(f"NVMe storage has {free_gib} GiB free; {minimum} GiB is required")
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        "--volume",
        f"{storage}:/models",
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


def _ensure_winnow_model(env: dict[str, str]) -> None:
    """Download (or re-verify) the pinned GGUF files with Winnow's own tool.

    ``scripts/download.py`` fetches the manifest's revision-pinned URLs and
    checks every file's size and SHA-256; the example's pins must match it.
    """

    image = env["WINNOW_IMAGE"]
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--volume",
            f"{env['WINNOW_MODEL_STORAGE_PATH']}:/models",
            "--entrypoint",
            "python3",
            image,
            "scripts/download.py",
            "--model-dir",
            "/models",
        ]
    )
    manifest = json.loads(
        _run(
            ["docker", "run", "--rm", "--entrypoint", "cat", image, "manifests/models.json"],
            capture=True,
        ).stdout
    )
    release = manifest["release"]
    pinned = SPEC["winnow"]["model"]
    shipped = {
        release[kind]["file"]: {"bytes": release[kind]["bytes"], "sha256": release[kind]["sha256"]}
        for kind in ("model", "projector")
    }
    if manifest.get("model_revision") != pinned["revision"] or shipped != pinned["files"]:
        raise SystemExit("winnow-inference manifest differs from example.json model pins")


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
        healthy = {
            pool: _healthy_replicas(metrics, pool)
            for pool in (DEEPSEEK_SERVED, QWEN_SERVED, WINNOW_SERVED)
        }
    except (KeyError, OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Kairyu readiness evidence is incomplete: {error}") from error
    if ready.get("status") != "ready" or models != set(PUBLIC_MODELS):
        raise SystemExit(f"Kairyu must serve exactly {PUBLIC_MODELS!r}, got {sorted(models)!r}")
    unhealthy = {pool: count for pool, count in healthy.items() if count != 1}
    if unhealthy:
        raise SystemExit(f"every pool must report 1 healthy replica, got {unhealthy!r}")


# Readiness checks grammar-constrained JSON in both thinking and non-thinking
# mode on every DP rank.
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


def deepseek_probe(*, thinking: bool) -> dict:
    payload: dict = {
        "model": DEEPSEEK_SERVED,
        "messages": [{"role": "user", "content": 'What is 17 * 19? Answer as JSON {"answer": n}.'}],
        "max_tokens": 8192,
        "response_format": _ANSWER_SCHEMA,
    }
    if thinking:
        payload["reasoning_effort"] = "high"
    return payload


def deepseek_answer_error(body: dict) -> str | None:
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


def _validate_deepseek(l1_url: str) -> None:
    # One request per DP rank and mode at once so every rank answers.
    requests = [thinking for thinking in (True, False) for _ in DP_RANK_GPU_IDS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:
        bodies = list(
            pool.map(
                lambda thinking: post_json(
                    f"{l1_url}/v1/chat/completions",
                    deepseek_probe(thinking=thinking),
                    timeout_s=900,
                ),
                requests,
            )
        )
    for thinking, body in zip(requests, bodies, strict=True):
        error = deepseek_answer_error(body)
        if error is not None:
            raise SystemExit(
                f"DeepSeek {'thinking' if thinking else 'chat'} JSON probe failed: {error}"
            )


def chat_probe(model: str) -> dict:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 17 * 19? Reply with the number only."}],
        "max_tokens": 2048,
    }
    if model == QWEN_SERVED:
        # Without an explicit mode the qwen3 parser returns the reply as reasoning.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def chat_answer_error(body: dict) -> str | None:
    try:
        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return f"malformed response: {str(body)[:200]}"
    if choice.get("finish_reason") != "stop":
        return f"finish_reason is {choice.get('finish_reason')!r}"
    if "323" not in content:
        return f"answer is {content[:200]!r}, expected 323"
    return None


def _validate_chat(l1_url: str, model: str) -> None:
    error = chat_answer_error(
        post_json(f"{l1_url}/v1/chat/completions", chat_probe(model), timeout_s=900)
    )
    if error is not None:
        raise SystemExit(f"{model} chat probe failed: {error}")


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
    """Winnow answers System One on its own L1 (the judge is not a public model)."""

    body = post_json(
        f"{l1_url}/v1/systemone",
        {"model": SPEC["systemone"]["upstream_model"], **SYSTEMONE_PROBE},
        timeout_s=300,
    )
    error = systemone_probe_error(body)
    if error is not None:
        raise SystemExit(f"Winnow System One probe failed: {error}")


def routed_request(content: str, *, model: str = ALWAYS_MODEL, **overrides) -> dict:
    """A chat request; the always-verified model unless ``model`` says otherwise."""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        # DeepSeek-V4.1's model card: max_tokens >= 256K.
        "max_tokens": 262144,
    }
    payload.update(overrides)
    return payload


def validate_verified_answer(api_url: str) -> None:
    body = post_json(
        f"{api_url}/v1/chat/completions",
        routed_request("Name the capital of France in one word."),
        timeout_s=1800,
    )
    try:
        content = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as error:
        raise SystemExit(f"verified-route probe failed: {str(body)[:200]}") from error
    if "paris" not in content.lower():
        raise SystemExit(f"verified-route probe failed: {content[:200]!r}")


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
    """Install the Reasoning Effort dropdown and check both models are offered."""

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
            "meta": {"description": "Reasoning effort for the think route."},
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
    _validate_deepseek(f"http://127.0.0.1:{env['DEEPSEEK_L1_PORT']}")
    _validate_chat(f"http://127.0.0.1:{env['QWEN_L1_PORT']}", QWEN_SERVED)
    _validate_chat(f"http://127.0.0.1:{env['WINNOW_L1_PORT']}", WINNOW_SERVED)
    _validate_systemone(f"http://127.0.0.1:{env['WINNOW_L1_PORT']}")
    validate_verified_answer(api_url)


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
    bind = env["PLAYGROUND_BIND_ADDRESS"]
    ui_host = _public_ui_host() if bind == "0.0.0.0" else bind
    env.setdefault("WEBUI_URL", f"http://{ui_host}:{env['CHAT_UI_PORT']}")
    _preflight(env)
    _ensure_deepseek_image(env)
    _ensure_qwen_image(env)
    _ensure_winnow_image(env)
    _ensure_model(
        env["DEEPSEEK_MODEL_STORAGE_PATH"],
        SPEC["deepseek"],
        env["DEEPSEEK_VLLM_IMAGE"],
        "DEEPSEEK_MODEL_SEED",
    )
    _ensure_model(
        env["QWEN_MODEL_STORAGE_PATH"],
        SPEC["qwen"],
        env["QWEN_VLLM_IMAGE"],
        "QWEN_MODEL_SEED",
    )
    _ensure_winnow_model(env)
    _compose(
        ["up", "--build", "--detach", "--wait", "--wait-timeout", "7200"],
        env=env,
    )
    validate_serving(env)
    provision_chat_ui(f"http://127.0.0.1:{env['CHAT_UI_PORT']}")
    print("\nEnvironment is ready.")
    print(f"Chat UI:     http://{ui_host}:{env['CHAT_UI_PORT']} (Open WebUI, no authentication)")
    print(f"Answer page: http://{ui_host}:{env['PLAYGROUND_PORT']} ({ALWAYS_MODEL})")
    print(f"OpenAI API:  http://{ui_host}:{env['PLAYGROUND_PORT']}/v1 (models {PUBLIC_MODELS})")


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
