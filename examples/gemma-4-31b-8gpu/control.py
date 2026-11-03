#!/usr/bin/env python3
"""One-command lifecycle for Gemma 4 31B chat and EmbeddingGemma 2 replicas on eight GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
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
PROJECT = SPEC["environment"].replace(".", "-")
ALLOCATION = SPEC["allocation"]
GPU_IDS: list[int] = [int(index) for index in ALLOCATION["gpu_ids"]]
CHAT_TP = int(ALLOCATION["chat"]["tensor_parallel_size"])
# Chat replica r owns the GPUs at position r; embedding replica e shares or
# owns one GPU. Service names follow the order: gemma-<r>, embed-<e>.
CHAT_REPLICA_GPU_IDS: list[list[int]] = [
    [int(index) for index in group] for group in ALLOCATION["chat"]["replica_gpu_ids"]
]
EMBED_REPLICA_GPU_IDS: list[int] = [
    int(index) for index in ALLOCATION["embedding"]["replica_gpu_ids"]
]
CHAT_SERVICES = [f"gemma-{replica}" for replica in range(len(CHAT_REPLICA_GPU_IDS))]
EMBED_SERVICES = [f"embed-{replica}" for replica in range(len(EMBED_REPLICA_GPU_IDS))]
L1_SERVICES = [*CHAT_SERVICES, *EMBED_SERVICES]
CHAT_MODEL = SPEC["model"]["served_name"]
EMBED_MODEL = SPEC["embedding"]["served_name"]
WEBUI = SPEC["webui"]
RETRIEVAL = WEBUI["retrieval"]
CODE_INTERPRETER = WEBUI["code_interpreter"]
# Open WebUI's data volume; the reranker lives in a subdirectory of it.
WEBUI_DATA = "/app/backend/data"
CHAT_UI_CONTAINER = f"{PROJECT}-chat-ui-1"


def _reranker_path() -> str:
    return f"{WEBUI_DATA}/{RETRIEVAL['models_dir']}/{RETRIEVAL['reranking']['slug']}"


def _check_allocation() -> None:
    chat_gpus = [index for group in CHAT_REPLICA_GPU_IDS for index in group]
    chat_util = float(ALLOCATION["chat"]["gpu_memory_utilization"])
    embed_util = float(ALLOCATION["embedding"]["gpu_memory_utilization"])
    per_gpu = {index: 0.0 for index in GPU_IDS}
    for index in chat_gpus:
        per_gpu[index] = per_gpu.get(index, 0.0) + chat_util
    for index in EMBED_REPLICA_GPU_IDS:
        per_gpu[index] = per_gpu.get(index, 0.0) + embed_util
    if (
        not CHAT_REPLICA_GPU_IDS
        or any(len(group) != CHAT_TP for group in CHAT_REPLICA_GPU_IDS)
        or len(set(chat_gpus)) != len(chat_gpus)
        # Two or more embedding replicas: one failure must not stop knowledge
        # bases (the replica pool moves the next batch to the other one).
        or len(EMBED_REPLICA_GPU_IDS) < 2
        or set(per_gpu) != set(GPU_IDS)
        or len(GPU_IDS) != int(SPEC["hardware"]["gpu_count"])
        or any(share > 0.98 for share in per_gpu.values())
    ):
        raise SystemExit(
            "example.json allocation is inconsistent: TP groups must be disjoint GPU "
            "sets of the declared size, at least two embedding replicas, only the "
            "declared GPUs, and at most 0.98 of any GPU's memory"
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
        "models": environment / "models",
        "webui": environment / "webui-data",
        "placement_log": environment / "placement-log",
        "cache": environment / "compile-cache",
    }
    for path in paths.values():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(f"cannot prepare NVMe storage {path}: {error}") from error
    return paths


def _json_compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def chat_ui_openai_api_configs() -> dict:
    # Connection 0 is OPENAI_API_BASE_URL (Kairyu). Only the chat model is
    # offered; Kairyu's model list also names the embedding model.
    return {"0": {"api_type": "responses", "model_ids": [CHAT_MODEL]}}


def _compose_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME"}
    }
    paths = _storage_paths()
    port = os.environ.get("CHAT_UI_PORT", str(WEBUI["port"]))
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": PROJECT,
            "MODEL_STORAGE_PATH": str(paths["models"]),
            "CACHE_PATH": str(paths["cache"]),
            "WEBUI_STORAGE_PATH": str(paths["webui"]),
            "PLACEMENT_LOG_PATH": str(paths["placement_log"]),
            "VLLM_IMAGE": os.environ.get("VLLM_IMAGE", SPEC["vllm"]["image"]),
            "OPEN_WEBUI_IMAGE": os.environ.get("OPEN_WEBUI_IMAGE", WEBUI["image"]),
            "CHAT_GPU_MEMORY_UTILIZATION": str(ALLOCATION["chat"]["gpu_memory_utilization"]),
            "EMBED_GPU_MEMORY_UTILIZATION": str(ALLOCATION["embedding"]["gpu_memory_utilization"]),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "CHAT_UI_PORT": port,
            "CHAT_UI_BIND_ADDRESS": os.environ.get("CHAT_UI_BIND_ADDRESS", WEBUI["bind"]),
            "WEBUI_URL": os.environ.get("WEBUI_URL", f"http://localhost:{port}"),
            "WEBUI_ADMIN_EMAIL": os.environ.get("WEBUI_ADMIN_EMAIL", ""),
            "WEBUI_ADMIN_PASSWORD": os.environ.get("WEBUI_ADMIN_PASSWORD", ""),
            "CHAT_UI_OPENAI_API_CONFIGS": _json_compact(chat_ui_openai_api_configs()),
            "CHAT_UI_DEFAULT_MODEL_PARAMS": _json_compact(WEBUI["default_params"]),
            "DOCLING_IMAGE": os.environ.get("DOCLING_IMAGE", RETRIEVAL["docling_image"]),
            "DOCLING_SERVER_URL": RETRIEVAL["docling_url"],
            "DOCLING_PARAMS": _json_compact(RETRIEVAL["docling_params"]),
            "RAG_OPENAI_API_BASE_URL": RETRIEVAL["embedding_api_url"],
            "RAG_EMBEDDING_BATCH_SIZE": str(RETRIEVAL["embedding_batch_size"]),
            "RAG_EMBEDDING_CONCURRENT_REQUESTS": str(RETRIEVAL["embedding_concurrent_requests"]),
            "RAG_EMBEDDING_QUERY_PREFIX": SPEC["embedding"]["query_prefix"],
            "RAG_EMBEDDING_CONTENT_PREFIX": SPEC["embedding"]["document_prefix"],
            "RAG_RERANKING_MODEL_PATH": _reranker_path(),
            "RAG_HYBRID_BM25_WEIGHT": str(RETRIEVAL["hybrid_bm25_weight"]),
            "RAG_TOP_K": str(RETRIEVAL["top_k"]),
            "RAG_TOP_K_RERANKER": str(RETRIEVAL["top_k_reranker"]),
            "CHUNK_SIZE": str(RETRIEVAL["chunk_size"]),
            "CHUNK_OVERLAP": str(RETRIEVAL["chunk_overlap"]),
            "CODE_INTERPRETER_ENGINE": CODE_INTERPRETER["engine"],
            "ENABLE_PYODIDE_FILE_PERSISTENCE": str(CODE_INTERPRETER["file_persistence"]).lower(),
            # A default feature applies only to models whose capabilities
            # include it, so the defaults carry Open WebUI v0.12.3's own
            # default capability set unchanged.
            "DEFAULT_MODEL_METADATA": _json_compact(
                {
                    "capabilities": CODE_INTERPRETER["capabilities"],
                    "defaultFeatureIds": CODE_INTERPRETER["default_features"],
                    # Files the code interpreter writes reach the user through
                    # this tool's download link (webui-file-download-tool.py).
                    "toolIds": CODE_INTERPRETER["default_tools"],
                }
            ),
        }
    )
    # Render-safe defaults for down/status/logs; `up` replaces them with each
    # replica's NUMA-local CPU set.
    for service in L1_SERVICES:
        key = f"{service.replace('-', '_').upper()}_CPUSET"
        env[key] = os.environ.get(key, "0")
    return env


def _compose_command(arguments: list[str]) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-directory",
        str(HERE),
        "--file",
        str(HERE / "compose.yaml"),
        *arguments,
    ]


def _compose(arguments: list[str], *, check: bool = True) -> None:
    _run(_compose_command(arguments), env=_compose_env(), check=check)


def compose_layout(config: dict) -> dict[str, list[str]]:
    """GPU ids per L1 service in a rendered compose file (``docker compose config``)."""

    layout: dict[str, list[str]] = {}
    for name, service in (config.get("services") or {}).items():
        devices = (
            ((service.get("deploy") or {}).get("resources") or {}).get("reservations") or {}
        ).get("devices") or []
        ids = [str(index) for device in devices for index in device.get("device_ids") or ()]
        if ids:
            layout[name] = ids
    return layout


def expected_layout() -> dict[str, list[str]]:
    layout = {
        service: [str(index) for index in group]
        for service, group in zip(CHAT_SERVICES, CHAT_REPLICA_GPU_IDS, strict=True)
    }
    layout.update(
        {
            service: [str(index)]
            for service, index in zip(EMBED_SERVICES, EMBED_REPLICA_GPU_IDS, strict=True)
        }
    )
    return layout


def _check_compose_layout(env: dict[str, str]) -> None:
    """compose.yaml must place exactly example.json's replicas on its GPUs."""

    rendered = _run(_compose_command(["config", "--format", "json"]), env=env, capture=True)
    config = json.loads(rendered.stdout)
    layout = compose_layout(config)
    if layout != expected_layout():
        raise SystemExit(
            f"compose.yaml places {layout!r}; example.json allocation is {expected_layout()!r}"
        )
    for service in CHAT_SERVICES:
        command = [str(part) for part in config["services"][service].get("command") or ()]
        if command[command.index("--tensor-parallel-size") + 1] != str(CHAT_TP):
            raise SystemExit(f"{service} must run TP{CHAT_TP} (example.json allocation)")


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


def _l1_running() -> bool:
    for service in L1_SERVICES:
        state = _run(
            ["docker", "inspect", "--format", "{{.State.Running}}", f"{PROJECT}-{service}-1"],
            capture=True,
            check=False,
        )
        if state.returncode == 0 and state.stdout.strip() == "true":
            return True
    return False


def _preflight(env: dict[str, str]) -> None:
    for executable in ("docker", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable is missing: {executable}")
    if not env["WEBUI_ADMIN_EMAIL"] or not env["WEBUI_ADMIN_PASSWORD"]:
        raise SystemExit(
            "set WEBUI_ADMIN_EMAIL and WEBUI_ADMIN_PASSWORD: the Chat UI is published on "
            "the host's address with sign-in, its first start creates this admin, and "
            "run.sh up signs in as the admin to check the Chat UI configuration"
        )
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).stdout
    rows = gpu_inventory(query)
    expected = SPEC["hardware"]
    missing = [index for index in GPU_IDS if index not in rows]
    if missing:
        raise SystemExit(f"GPUs {missing} are not present; this example uses GPUs {GPU_IDS}")
    ours = _l1_running()
    for index in GPU_IDS:
        row = rows[index]
        if row["name"] != expected["product"]:
            raise SystemExit(f"GPU {index} is {row['name']!r}; expected {expected['product']!r}")
        if row["memory_mib"] < expected["minimum_vram_mib"]:
            raise SystemExit(f"GPU {index} has insufficient VRAM")
        if row["compute_capability"] < expected["minimum_compute_capability"]:
            raise SystemExit(f"GPU {index} has insufficient compute capability")
        # Never evict another workload: the GPUs must be idle unless this
        # example's own L1 already holds them.
        if not ours and int(row["memory_used_mib"]) > int(expected["maximum_idle_used_mib"]):
            raise SystemExit(
                f"GPU {index} already has {row['memory_used_mib']} MiB in use; "
                "stop the workload holding it before starting this example"
            )
    if not ours:
        available = _host_memory_available_gib()
        required = int(expected["minimum_host_available_gib"])
        if available < required:
            raise SystemExit(f"host has {available} GiB available; the L1 needs {required} GiB")
    nodes = {index: numa_node(str(rows[index]["pci_bus_id"])) for index in GPU_IDS}
    for service, gpus in expected_layout().items():
        local = dict.fromkeys(nodes[int(index)] for index in gpus)
        env[f"{service.replace('-', '_').upper()}_CPUSET"] = ",".join(
            _node_cpulist(node) for node in local
        )
    layout = "; ".join(
        f"{service}=GPU {','.join(gpus)}" for service, gpus in expected_layout().items()
    )
    print(f"hardware: {len(GPU_IDS)} x {expected['product']}; {layout}", flush=True)


def _image_present(image: str) -> bool:
    inspected = _run(["docker", "image", "inspect", image], capture=True, check=False)
    return inspected.returncode == 0


def _ensure_vllm_image(env: dict[str, str]) -> None:
    """Pull the pinned upstream vLLM release (by digest) when it is absent."""

    image = env["VLLM_IMAGE"]
    pinned = {SPEC["vllm"]["image"], SPEC["vllm"]["fallback_image"]}
    if image not in pinned:
        raise SystemExit(f"VLLM_IMAGE must be one of the pinned releases {sorted(pinned)!r}")
    if not _image_present(image):
        print("vLLM image is absent; pulling the pinned upstream release", flush=True)
        _run(["docker", "pull", image])


_MODEL_PROGRAM = r"""
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, revision, slug, expected_tree, *rest = sys.argv[1:]
allow_patterns = json.loads(rest[0]) if rest else None
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

snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=allow_patterns,
                  token=os.environ.get('HF_TOKEN') or None)
rows, tree = inventory()
if tree != expected_tree:
    raise SystemExit(f'downloaded checkpoint tree mismatch: {tree}')
attestation.write_text(json.dumps({'schema_version': 1, 'repo': repo,
                                    'revision': revision, 'tree_sha256': tree,
                                    'files': rows}, sort_keys=True))
"""


def _attest_model(env: dict[str, str], volume: Path, model: dict, *extra: str) -> None:
    """Download ``model`` into ``volume`` at its pinned revision and check its tree hash."""

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
            *extra,
        ]
    )
    _run(command)


CHECKPOINTS = (SPEC["model"], SPEC["drafter"], SPEC["embedding"])


def _ensure_models(env: dict[str, str]) -> None:
    """The chat model, its MTP drafter and the embedding model, pinned and hashed."""

    volume = Path(env["MODEL_STORAGE_PATH"])
    absent = [
        model for model in CHECKPOINTS
        if not (volume / model["slug"] / ".kairyu-model-attestation.json").exists()
    ]
    if absent:
        free_gib = shutil.disk_usage(volume).free // (1024**3)
        minimum = int(SPEC["storage"]["minimum_free_gib"])
        if free_gib < minimum:
            raise SystemExit(f"NVMe storage has {free_gib} GiB free; {minimum} GiB is required")
    for model in CHECKPOINTS:
        _attest_model(env, volume, model)


def _ensure_reranker(env: dict[str, str]) -> None:
    """Fetch the Chat UI's reranker on the host side.

    Open WebUI runs with ``OFFLINE_MODE`` and loads it from its data volume,
    so a pinned, hashed copy must be there before it starts.
    """

    volume = Path(env["WEBUI_STORAGE_PATH"]) / RETRIEVAL["models_dir"]
    volume.mkdir(parents=True, exist_ok=True)
    model = RETRIEVAL["reranking"]
    _attest_model(env, volume, model, json.dumps(model["allow_patterns"]))


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


# --- Responses gate adapter (m11 D7 amendment 2026-10-17) ---------------------
# The gates are written against chat-shaped payloads and replies. These helpers
# send each payload to Kairyu's public /v1/responses instead (the API the Chat
# UI uses) and read the Responses reply back into the chat shape the gate
# checks.

# Responses fields a payload may already carry (the Chat UI filter's output
# has reasoning and store) pass through as they are.
_RESPONSES_PASSTHROUGH = frozenset(
    {
        "model",
        "stream",
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "metadata",
        "user",
        "reasoning",
        "store",
    }
)


def _responses_content(content: object, *, assistant: bool) -> object:
    if isinstance(content, str) or content is None:
        return content or ""
    parts = []
    for part in content:
        if part.get("type") == "text":
            kind = "output_text" if assistant else "input_text"
            parts.append({"type": kind, "text": part["text"]})
        elif part.get("type") == "image_url" and not assistant:
            image = part["image_url"]
            url = image["url"] if isinstance(image, dict) else image
            parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
        else:
            raise ValueError(f"chat content part {part.get('type')!r} has no Responses form")
    return parts


def _responses_tool(tool: dict) -> dict:
    function = tool["function"]
    flat = {"type": "function", "name": function["name"]}
    for key in ("description", "parameters", "strict"):
        if key in function:
            flat[key] = function[key]
    return flat


def responses_request(payload: dict) -> dict:
    """The /v1/responses body for one chat-shaped gate payload (stateless)."""

    body: dict = {"store": False}
    for key, value in payload.items():
        if key == "messages":
            continue
        if key in _RESPONSES_PASSTHROUGH:
            body[key] = value
        elif key in ("max_tokens", "max_completion_tokens"):
            body["max_output_tokens"] = value
        elif key == "reasoning_effort":
            body["reasoning"] = {"effort": value}
        elif key == "tools":
            body["tools"] = [_responses_tool(tool) for tool in value]
        elif key == "tool_choice":
            body["tool_choice"] = (
                {"type": "function", "name": value["function"]["name"]}
                if isinstance(value, dict)
                else value
            )
        elif key == "response_format":
            if value.get("type") == "json_schema":
                schema = value["json_schema"]
                body["text"] = {"format": {"type": "json_schema", **schema}}
            else:
                body["text"] = {"format": {"type": value["type"]}}
        elif key == "stream_options":
            continue  # a Responses stream always ends with usage
        else:
            raise ValueError(f"chat field {key!r} has no Responses form")
    items: list[dict] = []
    for message in payload["messages"]:
        role = message["role"]
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
            continue
        if message.get("reasoning_content"):
            items.append(
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": message["reasoning_content"]}],
                }
            )
        content = message.get("content")
        if content:
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": _responses_content(content, assistant=role == "assistant"),
                }
            )
        for call in message.get("tool_calls") or ():
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
            )
    body["input"] = items
    return body


def _finish_reason(response: dict, calls: list) -> str:
    if response.get("status") == "completed":
        return "tool_calls" if calls else "stop"
    if response.get("status") == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        return "content_filter" if reason == "content_filter" else "length"
    return str(response.get("status"))


def chat_view(response: dict) -> dict:
    """One Responses object read back as a chat completion (kairyu_* kept)."""

    text: list[str] = []
    reasoning: list[str] = []
    calls: list[dict] = []
    has_message = False
    for item in response.get("output") or ():
        kind = item.get("type")
        if kind == "message":
            has_message = True
            text += [
                part.get("text", "")
                for part in item.get("content") or ()
                if part.get("type") == "output_text"
            ]
        elif kind == "reasoning":
            parts = item.get("content") or item.get("summary") or ()
            reasoning += [p.get("text", "") for p in parts]
        elif kind == "function_call":
            calls.append(
                {
                    "id": item.get("call_id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments")},
                }
            )
    usage = response.get("usage") or {}
    chat_usage = {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_tokens_details": {
            "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens")
        },
        "completion_tokens_details": {
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        },
    }
    return {
        "id": response.get("id"),
        "object": "chat.completion",
        "created": response.get("created_at"),
        "model": response.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(text) if has_message else None,
                    "reasoning_content": "".join(reasoning) or None,
                    "tool_calls": calls or None,
                },
                "finish_reason": _finish_reason(response, calls),
            }
        ],
        "usage": chat_usage,
        **{key: value for key, value in response.items() if key.startswith("kairyu_")},
    }


def chat_stream_view(sse_text: str) -> str:
    """A Responses SSE body read back as chat SSE chunks ending in [DONE]."""

    chunks: list[dict] = []
    indexes: dict[int, int] = {}
    streamed_arguments: set[int] = set()

    def delta(**fields) -> None:
        chunks.append({"choices": [{"index": 0, "delta": fields, "finish_reason": None}]})

    for line in sse_text.splitlines():
        if not line.startswith("data:") or not line[len("data:") :].strip():
            continue
        event = json.loads(line[len("data:") :])
        kind = event.get("type")
        if kind == "response.output_text.delta":
            delta(content=event["delta"])
        elif kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            delta(reasoning_content=event["delta"])
        elif kind == "response.output_item.added" and event["item"].get("type") == "function_call":
            index = indexes.setdefault(event["output_index"], len(indexes))
            item = event["item"]
            delta(
                tool_calls=[
                    {
                        "index": index,
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {"name": item.get("name"), "arguments": ""},
                    }
                ]
            )
        elif kind == "response.function_call_arguments.delta":
            index = indexes[event["output_index"]]
            streamed_arguments.add(index)
            delta(tool_calls=[{"index": index, "function": {"arguments": event["delta"]}}])
        elif kind == "response.output_item.done" and event["item"].get("type") == "function_call":
            index = indexes[event["output_index"]]
            if index not in streamed_arguments:
                arguments = event["item"].get("arguments") or ""
                delta(tool_calls=[{"index": index, "function": {"arguments": arguments}}])
        elif kind in ("response.completed", "response.incomplete"):
            view = chat_view(event["response"])
            extensions = {k: v for k, v in view.items() if k.startswith("kairyu_")}
            finish = view["choices"][0]["finish_reason"]
            chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            chunks.append({"choices": [], "usage": view["usage"], **extensions})
        elif kind in ("error", "response.failed"):
            error = event if kind == "error" else (event.get("response") or {}).get("error")
            chunks.append({"error": error})
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def post_responses(api_url: str, payload: dict, *, timeout_s: float) -> dict:
    """POST one chat-shaped payload to /v1/responses; return the reply as chat."""
    reply = post_json(f"{api_url}/v1/responses", responses_request(payload), timeout_s=timeout_s)
    return chat_view(reply)


def healthy_replicas(metrics: str, pool: str) -> int | None:
    pattern = re.compile(
        r'^kairyu_pool_replicas\{pool="' + re.escape(pool) + r'",state="healthy"\} (\S+)$',
        re.MULTILINE,
    )
    match = pattern.search(metrics)
    return int(float(match.group(1))) if match else None


def validate_ready(api_url: str) -> None:
    try:
        ready = _json_url(f"{api_url}/readyz")
        models = {row["id"] for row in _json_url(f"{api_url}/v1/models")["data"]}
        metrics = _text_url(f"{api_url}/metrics")
    except (KeyError, OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Kairyu readiness evidence is incomplete: {error}") from error
    if ready.get("status") != "ready" or models != {CHAT_MODEL, EMBED_MODEL}:
        raise SystemExit(
            f"Kairyu public models must be exactly [{CHAT_MODEL!r}, {EMBED_MODEL!r}], "
            f"got {sorted(models)!r}"
        )
    for pool, expected in ((CHAT_MODEL, len(CHAT_SERVICES)), (EMBED_MODEL, len(EMBED_SERVICES))):
        healthy = healthy_replicas(metrics, pool)
        if healthy != expected:
            raise SystemExit(
                f"Kairyu pool {pool!r} must report {expected} healthy replicas, got {healthy!r}"
            )


def arithmetic_answer_error(body: dict, *, logprobs: bool = True) -> str | None:
    """Reject a probe that is not exactly ``323`` with finite log-probabilities."""

    try:
        choice = body["choices"][0]
        content = choice["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return f"malformed response: {str(body)[:200]}"
    if choice.get("finish_reason") != "stop":
        return f"finish_reason is {choice.get('finish_reason')!r}"
    if not isinstance(content, str) or content.strip() != "323":
        return f"answer is {content!r}, expected '323'"
    if not logprobs:
        return None
    tokens = ((choice.get("logprobs") or {}).get("content")) or []
    if not tokens:
        return "no log-probabilities were returned"
    for token in tokens:
        value = token.get("logprob")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return f"non-finite log-probability {value!r}"
    return None


def arithmetic_probe(*, logprobs: bool = True, **overrides) -> dict:
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": "What is 17 * 19? Reply with only the integer."}],
        "max_tokens": 8192,
    }
    if logprobs:
        payload.update(logprobs=True, top_logprobs=3)
    payload.update(overrides)
    return payload


def _validate_arithmetic(api_url: str) -> None:
    # One request per chat replica and thinking mode at once: least-outstanding
    # placement gives every replica one of each. /v1/responses (the Chat UI's
    # API) has no token log-probabilities, so this checks the exact answer and
    # finish; `verify.sh l1` reads finite log-probabilities from vLLM directly.
    variants = [{}, {"reasoning_effort": "none"}]
    requests = [variant for variant in variants for _ in CHAT_SERVICES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CHAT_SERVICES)) as pool:
        bodies = list(
            pool.map(
                lambda overrides: post_responses(
                    api_url,
                    arithmetic_probe(logprobs=False, **overrides),
                    timeout_s=900,
                ),
                requests,
            )
        )
    for overrides, body in zip(requests, bodies, strict=True):
        error = arithmetic_answer_error(body, logprobs=False)
        if error is not None:
            raise SystemExit(f"arithmetic probe {overrides or 'default'} failed: {error}")


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}


def tool_probe() -> dict:
    return {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an agent operating a computer shell. Every response "
                    "MUST include at least one bash tool call; never answer in "
                    "plain text."
                ),
            },
            {"role": "user", "content": "List the files in the current directory."},
        ],
        "tools": [BASH_TOOL],
        "max_tokens": 8192,
    }


def tool_call_error(body: dict) -> str | None:
    try:
        choice = body["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        arguments = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return f"malformed tool reply: {error}"
    if (
        choice.get("finish_reason") != "tool_calls"
        or not calls
        or calls[0]["function"].get("name") != "bash"
        or not isinstance(arguments, dict)
        or not isinstance(arguments.get("command"), str)
        or not arguments["command"]
    ):
        return (
            "no executable bash tool call (finish_reason="
            f"{choice.get('finish_reason')!r}, tool_calls={json.dumps(calls)[:300]})"
        )
    return None


def validate_tool_calling(api_url: str) -> None:
    try:
        body = post_responses(api_url, tool_probe(), timeout_s=600)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"tool-calling probe failed: {error}") from error
    error = tool_call_error(body)
    if error is not None:
        raise SystemExit(f"tool-calling probe: {error}")


# A 64x64 solid-red PNG.
PROBE_IMAGE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQkAAAgEsItz/fMY"
    "xgi+hcEKLNO+FgEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQGB"
    "ywLzk8EPlvGqjQAAAABJRU5ErkJggg=="
)


def image_request(text: str, *, max_tokens: int) -> dict:
    return {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{PROBE_IMAGE_PNG_BASE64}"},
                    },
                    {"type": "text", "text": text},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }


def validate_vision(api_url: str) -> None:
    payload = image_request(
        "What single color fills this image? Answer with one word.", max_tokens=8192
    )
    try:
        body = post_responses(api_url, payload, timeout_s=900)
        message = body["choices"][0]["message"]
        content = message.get("content")
    except (KeyError, IndexError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"image probe failed: {error}") from error
    if (
        not isinstance(content, str)
        or not re.search(r"\bred\b", content, re.IGNORECASE)
        or body["choices"][0].get("finish_reason") != "stop"
    ):
        raise SystemExit(
            f"image probe: the API returned no completed red-image answer ({message!r})"
        )
    print(f"image probe: {content.strip()[:80]!r}", flush=True)


def embedding_vectors_error(body: dict, count: int) -> str | None:
    """Every vector: the model's width, finite values, unit length (L2-normalized)."""

    dimensions = int(SPEC["embedding"]["dimensions"])
    try:
        rows = sorted(body["data"], key=lambda row: row["index"])
        vectors = [row["embedding"] for row in rows]
        prompt_tokens = body["usage"]["prompt_tokens"]
    except (KeyError, TypeError) as error:
        return f"malformed embeddings reply: {error}"
    if body.get("model") != EMBED_MODEL or len(vectors) != count:
        return f"reply has model {body.get('model')!r} and {len(vectors)} vectors"
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return f"usage.prompt_tokens is {prompt_tokens!r}"
    for vector in vectors:
        if len(vector) != dimensions or not all(math.isfinite(value) for value in vector):
            return f"a vector has {len(vector)} values or a non-finite value"
        norm = math.sqrt(sum(value * value for value in vector))
        if abs(norm - 1.0) > 1e-3:
            return f"a vector has norm {norm:.5f}, expected 1"
    return None


def embedding_inputs() -> list[str]:
    spec = SPEC["embedding"]
    return [
        spec["query_prefix"] + "KR-7319という試料は何度で保存しますか。",
        spec["document_prefix"] + "試料KR-7319はマイナス四十二度の冷凍庫で保管する。",
    ]


def validate_embeddings(api_url: str) -> None:
    inputs = embedding_inputs()
    try:
        body = post_json(
            f"{api_url}/v1/embeddings", {"model": EMBED_MODEL, "input": inputs}, timeout_s=120
        )
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"embedding probe failed: {error}") from error
    error = embedding_vectors_error(body, len(inputs))
    if error is not None:
        raise SystemExit(f"embedding probe: {error}")


def validate_serving(api_url: str) -> None:
    """Readiness: both pools, exact answers on every chat replica, tools, image, embeddings."""

    validate_ready(api_url)
    _validate_arithmetic(api_url)
    validate_tool_calling(api_url)
    validate_vision(api_url)
    validate_embeddings(api_url)


# --- Chat UI provisioning and fail-closed checks -------------------------------

CHAT_UI_EFFORT_FILTER_ID = "reasoning_effort"
CHAT_UI_FILE_TOOL_ID = "file_download"
CHAT_UI_EFFORT_LEVELS: list[str] = list(WEBUI["reasoning_effort_levels"])


def webui_api(
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


def chat_ui_admin_token(ui_url: str) -> str:
    """Sign in as the admin that the Chat UI's first start created."""

    signin = webui_api(
        ui_url,
        "/api/v1/auths/signin",
        payload={
            "email": os.environ.get("WEBUI_ADMIN_EMAIL", ""),
            "password": os.environ.get("WEBUI_ADMIN_PASSWORD", ""),
        },
    )
    if not isinstance(signin, dict) or not signin.get("token"):
        raise SystemExit("Chat UI admin signin did not return a session token")
    if signin.get("role") != "admin":
        raise SystemExit("WEBUI_ADMIN_EMAIL must be the Chat UI's admin user")
    return signin["token"]


def _provision_chat_ui_file_download(ui_url: str, token: str) -> None:
    """Install the file-download tool the model's default toolIds name, and check it."""

    if CODE_INTERPRETER["default_tools"] != [CHAT_UI_FILE_TOOL_ID]:
        raise SystemExit(
            f"example.json default_tools must be [{CHAT_UI_FILE_TOOL_ID!r}], "
            f"got {CODE_INTERPRETER['default_tools']!r}"
        )
    source = (HERE / "webui-file-download-tool.py").read_text(encoding="utf-8")
    body = {
        "id": CHAT_UI_FILE_TOOL_ID,
        "name": "File download",
        "content": source,
        "meta": {"description": "Give the user a download link for a file the code wrote."},
        # Every signed-in user's chats carry this default tool.
        "access_grants": [{"principal_type": "user", "principal_id": "*", "permission": "read"}],
    }
    listed = webui_api(ui_url, "/api/v1/tools/", token=token)
    if any(row.get("id") == CHAT_UI_FILE_TOOL_ID for row in listed):
        webui_api(
            ui_url, f"/api/v1/tools/id/{CHAT_UI_FILE_TOOL_ID}/update", token=token, payload=body
        )
    else:
        webui_api(ui_url, "/api/v1/tools/create", token=token, payload=body)
    exported = webui_api(ui_url, "/api/v1/tools/export", token=token)
    installed = next((row for row in exported if row.get("id") == CHAT_UI_FILE_TOOL_ID), None)
    if installed is None or installed.get("content") != source:
        raise SystemExit("Chat UI file-download tool is not installed with this example's source")
    if not any(spec.get("name") == "provide_download" for spec in installed.get("specs") or ()):
        raise SystemExit("Chat UI file-download tool does not expose provide_download")
    if not any(
        (grant.get("principal_type"), grant.get("principal_id"), grant.get("permission"))
        == ("user", "*", "read")
        for grant in installed.get("access_grants") or ()
    ):
        raise SystemExit("Chat UI file-download tool is not readable by every signed-in user")


def _check_chat_ui_code_interpreter(ui_url: str, token: str) -> None:
    """Word / Excel / PowerPoint editing needs the browser code interpreter, on by default."""

    config = webui_api(ui_url, "/api/config", token=token)
    features = config.get("features") or {}
    code = config.get("code") or {}
    models = webui_api(ui_url, "/api/models", token=token).get("data") or []
    model = next((row for row in models if row.get("id") == CHAT_MODEL), {})
    meta = (model.get("info") or {}).get("meta") or {}
    observed = {
        "enable_code_interpreter": features.get("enable_code_interpreter"),
        "interpreter_engine": code.get("interpreter_engine"),
        "enable_pyodide_file_persistence": features.get("enable_pyodide_file_persistence"),
        "default_features": sorted(meta.get("defaultFeatureIds") or []),
        "default_tools": sorted(meta.get("toolIds") or []),
        "capabilities": meta.get("capabilities"),
    }
    expected = {
        "enable_code_interpreter": True,
        "interpreter_engine": CODE_INTERPRETER["engine"],
        "enable_pyodide_file_persistence": CODE_INTERPRETER["file_persistence"],
        "default_features": sorted(CODE_INTERPRETER["default_features"]),
        "default_tools": sorted(CODE_INTERPRETER["default_tools"]),
        "capabilities": CODE_INTERPRETER["capabilities"],
    }
    if observed != expected:
        raise SystemExit(
            f"Chat UI code interpreter differs from example.json: {observed!r}; "
            f"expected {expected!r}"
        )


def _check_chat_ui_connection(ui_url: str, token: str) -> None:
    """Responses to Kairyu (m11 D7 amendment), and only the chat model offered."""

    config = webui_api(ui_url, "/openai/config", token=token)
    connection = (config.get("OPENAI_API_CONFIGS") or {}).get("0") or {}
    expected = chat_ui_openai_api_configs()["0"]
    observed = {key: connection.get(key) for key in expected}
    if observed != expected:
        raise SystemExit(f"Chat UI connection 0 is {observed!r}; expected {expected!r}")
    listed = webui_api(ui_url, "/api/models", token=token).get("data") or []
    models = {row.get("id") for row in listed}
    if models != {CHAT_MODEL}:
        raise SystemExit(f"Chat UI must offer exactly [{CHAT_MODEL!r}], offers {sorted(models)!r}")


def expected_retrieval() -> dict:
    return {
        "CONTENT_EXTRACTION_ENGINE": "docling",
        "DOCLING_SERVER_URL": RETRIEVAL["docling_url"],
        "DOCLING_PARAMS": RETRIEVAL["docling_params"],
        "RAG_EMBEDDING_ENGINE": "openai",
        "RAG_EMBEDDING_MODEL": EMBED_MODEL,
        "RAG_OPENAI_API_BASE_URL": RETRIEVAL["embedding_api_url"],
        "RAG_EMBEDDING_BATCH_SIZE": RETRIEVAL["embedding_batch_size"],
        "ENABLE_ASYNC_EMBEDDING": True,
        "RAG_EMBEDDING_CONCURRENT_REQUESTS": RETRIEVAL["embedding_concurrent_requests"],
        "RAG_RERANKING_ENGINE": "",
        "RAG_RERANKING_MODEL": _reranker_path(),
        "ENABLE_RAG_HYBRID_SEARCH": True,
        "HYBRID_BM25_WEIGHT": RETRIEVAL["hybrid_bm25_weight"],
        "TOP_K": RETRIEVAL["top_k"],
        "TOP_K_RERANKER": RETRIEVAL["top_k_reranker"],
        "CHUNK_SIZE": RETRIEVAL["chunk_size"],
        "CHUNK_OVERLAP": RETRIEVAL["chunk_overlap"],
    }


def _check_chat_ui_retrieval(ui_url: str, token: str) -> None:
    """Knowledge bases: Docling extraction, EmbeddingGemma 2 through Kairyu, the pinned reranker.

    Every document and query embedding goes to Kairyu's /v1/embeddings
    (the embedding replica pool), never to an engine inside the Chat UI. BM25
    stays off: Open WebUI splits BM25 terms on whitespace, which cannot
    tokenize Japanese.
    """

    config = webui_api(ui_url, "/api/v1/retrieval/config", token=token)
    embedding = webui_api(ui_url, "/api/v1/retrieval/embedding", token=token)
    observed = {
        "CONTENT_EXTRACTION_ENGINE": config.get("CONTENT_EXTRACTION_ENGINE"),
        "DOCLING_SERVER_URL": config.get("DOCLING_SERVER_URL"),
        "DOCLING_PARAMS": config.get("DOCLING_PARAMS"),
        "RAG_EMBEDDING_ENGINE": embedding.get("RAG_EMBEDDING_ENGINE"),
        "RAG_EMBEDDING_MODEL": embedding.get("RAG_EMBEDDING_MODEL"),
        "RAG_OPENAI_API_BASE_URL": (embedding.get("openai_config") or {}).get("url"),
        "RAG_EMBEDDING_BATCH_SIZE": embedding.get("RAG_EMBEDDING_BATCH_SIZE"),
        "ENABLE_ASYNC_EMBEDDING": embedding.get("ENABLE_ASYNC_EMBEDDING"),
        "RAG_EMBEDDING_CONCURRENT_REQUESTS": embedding.get("RAG_EMBEDDING_CONCURRENT_REQUESTS"),
        "RAG_RERANKING_ENGINE": config.get("RAG_RERANKING_ENGINE"),
        "RAG_RERANKING_MODEL": config.get("RAG_RERANKING_MODEL"),
        "ENABLE_RAG_HYBRID_SEARCH": config.get("ENABLE_RAG_HYBRID_SEARCH"),
        "HYBRID_BM25_WEIGHT": config.get("HYBRID_BM25_WEIGHT"),
        "TOP_K": config.get("TOP_K"),
        "TOP_K_RERANKER": config.get("TOP_K_RERANKER"),
        "CHUNK_SIZE": config.get("CHUNK_SIZE"),
        "CHUNK_OVERLAP": config.get("CHUNK_OVERLAP"),
    }
    expected = expected_retrieval()
    wrong = {key: value for key, value in observed.items() if value != expected[key]}
    if wrong:
        raise SystemExit(
            f"Chat UI retrieval configuration differs from example.json: {wrong!r}; "
            f"expected { ({key: expected[key] for key in wrong})!r}"
        )
    _check_chat_ui_prefixes()


def chat_ui_environment() -> dict[str, str]:
    inspected = _run(
        ["docker", "inspect", "--format", "{{json .Config.Env}}", CHAT_UI_CONTAINER],
        capture=True,
    )
    return dict(item.split("=", 1) for item in json.loads(inspected.stdout) if "=" in item)


def _check_chat_ui_prefixes() -> None:
    """EmbeddingGemma 2's query and document instructions (no API reads them back)."""

    environment = chat_ui_environment()
    expected = {
        "RAG_EMBEDDING_QUERY_PREFIX": SPEC["embedding"]["query_prefix"],
        "RAG_EMBEDDING_CONTENT_PREFIX": SPEC["embedding"]["document_prefix"],
    }
    observed = {key: environment.get(key) for key in expected}
    if observed != expected or "RAG_EMBEDDING_PREFIX_FIELD_NAME" in environment:
        raise SystemExit(
            f"Chat UI embedding prefixes are {observed!r}; expected {expected!r} prepended "
            "to the text (no RAG_EMBEDDING_PREFIX_FIELD_NAME)"
        )


def provision_chat_ui(ui_url: str) -> None:
    """Install the thinking dropdown and the download tool; check every Chat UI setting."""

    filter_source = (HERE / "webui-responses-filter.py").read_text(encoding="utf-8")
    base = f"/api/v1/functions/id/{CHAT_UI_EFFORT_FILTER_ID}"
    try:
        token = chat_ui_admin_token(ui_url)
        listed = webui_api(ui_url, "/api/v1/functions/", token=token)
        existing = next((row for row in listed if row.get("id") == CHAT_UI_EFFORT_FILTER_ID), None)
        body = {
            "id": CHAT_UI_EFFORT_FILTER_ID,
            "name": "Thinking",
            "content": filter_source,
            "meta": {"description": "Turn Gemma 4 thinking off for a chat from a dropdown."},
        }
        if existing is None:
            state = webui_api(ui_url, "/api/v1/functions/create", token=token, payload=body)
        else:
            updated = webui_api(ui_url, f"{base}/update", token=token, payload=body)
            state = {**existing, **(updated or {})}
        # The toggle endpoints flip state, so call them only while a flag is off.
        if not state.get("is_active"):
            state = webui_api(ui_url, f"{base}/toggle", token=token, payload={})
        if not state.get("is_global"):
            state = webui_api(ui_url, f"{base}/toggle/global", token=token, payload={})
        if not (state.get("is_active") and state.get("is_global")):
            raise SystemExit("Chat UI thinking selector could not be activated globally")
        _check_chat_ui_connection(ui_url, token)
        _check_chat_ui_retrieval(ui_url, token)
        _provision_chat_ui_file_download(ui_url, token)
        _check_chat_ui_code_interpreter(ui_url, token)
        spec = webui_api(ui_url, f"{base}/valves/user/spec", token=token)
        enum = spec.get("properties", {}).get("reasoning_effort", {}).get("enum")
    except (KeyError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Chat UI provisioning failed: {error}") from error
    if enum != CHAT_UI_EFFORT_LEVELS:
        raise SystemExit(
            f"Chat UI thinking selector must expose exactly {CHAT_UI_EFFORT_LEVELS!r}, got {enum!r}"
        )


_NO_PUBLIC_HOST = "cannot discover an externally reachable host; set PUBLIC_HOST"


def public_host() -> str:
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


def chat_ui_local_url(env: dict[str, str]) -> str:
    bind = env["CHAT_UI_BIND_ADDRESS"]
    return f"http://{'127.0.0.1' if bind == '0.0.0.0' else bind}:{env['CHAT_UI_PORT']}"


def up() -> None:
    env = _compose_env()
    bind = env["CHAT_UI_BIND_ADDRESS"]
    ui_host = public_host() if bind == "0.0.0.0" else bind
    if "WEBUI_URL" not in os.environ:
        env["WEBUI_URL"] = f"http://{ui_host}:{env['CHAT_UI_PORT']}"
    _preflight(env)
    _check_compose_layout(env)
    _ensure_vllm_image(env)
    _ensure_models(env)
    _ensure_reranker(env)
    _run(
        _compose_command(["up", "--build", "--detach", "--wait", "--wait-timeout", "7200"]),
        env=env,
    )
    api_url = f"http://127.0.0.1:{env['API_PORT']}"
    validate_serving(api_url)
    provision_chat_ui(chat_ui_local_url(env))
    print("\nEnvironment is ready.")
    print(f"OpenAI API: {api_url}/v1 (chat {CHAT_MODEL}, embeddings {EMBED_MODEL})")
    print(f"Chat UI:    {env['WEBUI_URL']} (sign in as WEBUI_ADMIN_EMAIL)")
    print(
        f"Chat model: {CHAT_MODEL} ({len(CHAT_SERVICES)} TP{CHAT_TP} replicas; text + image "
        f"input; thinking on); embeddings: {EMBED_MODEL} ({len(EMBED_SERVICES)} replicas)"
    )
    print(
        "Thinking: Chat Controls -> Valves -> Thinking "
        f"({'/'.join(CHAT_UI_EFFORT_LEVELS)}; none turns it off)"
    )


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
