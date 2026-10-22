#!/usr/bin/env python3
"""One-command lifecycle for GLM-5.3-Flash as one DP6 / EP6 replica on six GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
ROOT = HERE.parents[1]
ALLOCATION = SPEC["allocation"]
GPU_IDS: list[int] = [int(index) for index in ALLOCATION["gpu_ids"]]
# DP rank r owns the TP group at positions [r * TP, (r + 1) * TP) of the
# container's device list (vLLM's DP-major local rank order).
DP_RANK_GPU_IDS: list[list[int]] = [list(group) for group in ALLOCATION["dp_rank_gpu_ids"]]
PROJECT = SPEC["environment"].replace(".", "-")
L1_CONTAINER = f"{PROJECT}-glm-1"

# The Chat UI is the shared L4 workspace (deploy/workspace/README.md); this
# example owns only its values, checked here at import.
sys.path.append(str(ROOT / "deploy" / "workspace"))
import kairyu_workspace as workspace  # noqa: E402

WORKSPACE = workspace.load(SPEC["workspace"])
COMPOSE_FILE_ARGS = ("--file", str(HERE / "compose.yaml"), "--file", str(workspace.COMPOSE_FILE))


def _check_allocation() -> None:
    tp = int(ALLOCATION["tensor_parallel_size"])
    dp = int(ALLOCATION["data_parallel_size"])
    flat = [index for group in DP_RANK_GPU_IDS for index in group]
    if (
        len(DP_RANK_GPU_IDS) != dp
        or any(len(group) != tp for group in DP_RANK_GPU_IDS)
        or flat != GPU_IDS
        or int(ALLOCATION["expert_parallel_size"]) != tp * dp
        or len(GPU_IDS) != int(SPEC["hardware"]["gpu_count"])
    ):
        raise SystemExit("example.json allocation is inconsistent (TP x DP must tile gpu_ids)")


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
        # Shared by every example's workspace: the retrieval models are
        # pinned by the profile, attested by hash and mounted read-only.
        "workspace_models": environment.parent / "workspace-models",
        "placement_log": environment / "placement-log",
        "cache": environment / "compile-cache",
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
            "GLM_MODEL_STORAGE_PATH": str(paths["models"]),
            "GLM_CACHE_PATH": str(paths["cache"]),
            "PLACEMENT_LOG_PATH": str(paths["placement_log"]),
            "GLM_VLLM_IMAGE": os.environ.get("GLM_VLLM_IMAGE", SPEC["vllm"]["image"]),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "API_BIND_ADDRESS": os.environ.get("API_BIND_ADDRESS", "0.0.0.0"),
            # Render-safe default for down/status/logs; `up` replaces it with
            # the GPUs' NUMA-local CPU sets.
            "GLM_CPUSET": os.environ.get("GLM_CPUSET", "0"),
        }
    )
    env.update(
        workspace.compose_env(WORKSPACE, data=paths["webui"], models=paths["workspace_models"])
    )
    return env


def _compose(arguments: list[str], *, env: dict[str, str] | None = None) -> None:
    _run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(HERE),
            *COMPOSE_FILE_ARGS,
            *arguments,
        ],
        env=env or _compose_env(),
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
    """NUMA node per DP rank; a TP group (e.g. the TP2 tuning candidate) must sit on one node.

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


def _l1_running() -> bool:
    state = _run(
        ["docker", "inspect", "--format", "{{.State.Running}}", L1_CONTAINER],
        capture=True,
        check=False,
    )
    return state.returncode == 0 and state.stdout.strip() == "true"


def _preflight(env: dict[str, str]) -> None:
    for executable in ("docker", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable is missing: {executable}")
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
    nodes = {index: numa_node(str(rows[index]["pci_bus_id"])) for index in GPU_IDS}
    layout = dp_rank_numa_layout(nodes, DP_RANK_GPU_IDS)
    env["GLM_CPUSET"] = ",".join(_node_cpulist(node) for node in dict.fromkeys(layout))
    ranks = "; ".join(
        f"DP{rank}=GPU {','.join(map(str, group))} (NUMA {layout[rank]})"
        for rank, group in enumerate(DP_RANK_GPU_IDS)
    )
    print(f"hardware: {len(GPU_IDS)} x {expected['product']}; {ranks}", flush=True)


def _image_field(image: str, template: str) -> str | None:
    inspected = _run(
        ["docker", "image", "inspect", "--format", template, image], capture=True, check=False
    )
    return inspected.stdout.strip() if inspected.returncode == 0 else None


def vllm_image_matches(image: str) -> bool:
    """The image is the pinned registry digest. An image ID is not compared: the classic
    store reports the config digest, the containerd store the manifest-list digest."""

    digests = json.loads(_image_field(image, "{{json .RepoDigests}}") or "null") or []
    return SPEC["vllm"]["repo_digest"] in digests


def _ensure_vllm_image(env: dict[str, str]) -> None:
    image = env["GLM_VLLM_IMAGE"]
    if _image_field(image, "{{.Id}}") is None:
        _run(["docker", "pull", image])
    if not vllm_image_matches(image):
        raise SystemExit(
            f"vLLM image {image} is not {SPEC['vllm']['repo_digest']} "
            f"(the stock {SPEC['vllm']['release']} image this example pins)"
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


def _seed_model(target: Path) -> bool:
    """Hard-link an already downloaded checkpoint instead of fetching 306 GiB.

    ``GLM_MODEL_SEED`` names a local copy of the same checkpoint on the same
    filesystem. Nothing is trusted from it: the copy is re-hashed against this
    example's pinned tree before it is served.
    """

    seed = os.environ.get("GLM_MODEL_SEED")
    if not seed or target.exists():
        return False
    source = Path(seed).resolve()
    if not (source / "config.json").is_file():
        raise SystemExit(f"GLM_MODEL_SEED has no checkpoint: {source}")
    _run(["cp", "-al", str(source), str(target)])
    (target / ".kairyu-model-attestation.json").unlink(missing_ok=True)
    return True


def _ensure_model(env: dict[str, str]) -> None:
    model = SPEC["model"]
    target = Path(env["GLM_MODEL_STORAGE_PATH"]) / model["slug"]
    seeded = _seed_model(target)
    if not (target / ".kairyu-model-attestation.json").exists() and not seeded:
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
        f"{env['GLM_MODEL_STORAGE_PATH']}:/models",
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
            env["GLM_VLLM_IMAGE"],
            "-c",
            _MODEL_PROGRAM,
            model["repo"],
            model["revision"],
            model["slug"],
            model["tree_sha256"],
        ]
    )
    _run(command)


def api_check_url(env) -> str:
    """Where this host reaches the API: loopback when it listens on every interface."""
    bind = env.get("API_BIND_ADDRESS", "0.0.0.0")
    host = "127.0.0.1" if bind == "0.0.0.0" else bind
    return f"http://{host}:{env.get('API_PORT', SPEC['api_port'])}"


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
# The example gates are written against chat-shaped payloads and replies. These
# helpers send each payload to Kairyu's public /v1/responses instead (the API
# the Chat UI uses) and read the Responses reply back into the chat shape the
# gate checks, so a gate's assertions stay as they were.

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
            parts = item.get("content") or ()
            text += [p.get("text", "") for p in parts if p.get("type") == "output_text"]
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
        **{key: value for key, value in usage.items() if key.startswith("orchestration_")},
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
    """POST a chat-shaped payload to Kairyu's /v1/responses; the reply in chat shape."""
    body = post_json(f"{api_url}/v1/responses", responses_request(payload), timeout_s=timeout_s)
    return chat_view(body)


def _healthy_replicas(metrics: str, pool: str) -> int | None:
    pattern = re.compile(
        r'^kairyu_pool_replicas\{pool="' + re.escape(pool) + r'",state="healthy"\} (\S+)$',
        re.MULTILINE,
    )
    match = pattern.search(metrics)
    return int(float(match.group(1))) if match else None


def validate_ready(api_url: str) -> None:
    served = SPEC["model"]["served_name"]
    try:
        ready = _json_url(f"{api_url}/readyz")
        models = {row["id"] for row in _json_url(f"{api_url}/v1/models")["data"]}
        healthy = _healthy_replicas(_text_url(f"{api_url}/metrics"), served)
    except (KeyError, OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Kairyu readiness evidence is incomplete: {error}") from error
    if ready.get("status") != "ready" or models != {served}:
        raise SystemExit(
            f"Kairyu public model inventory must be exactly [{served!r}], got {sorted(models)!r}"
        )
    if healthy != 1:
        raise SystemExit(f"Kairyu pool {served!r} must report 1 healthy replica, got {healthy!r}")


def arithmetic_answer_error(body: dict, *, logprobs: bool = True) -> str | None:
    """Reject a probe that is not exactly ``323`` with finite log-probabilities.

    A first correct answer is not evidence on a new SM120 kernel path, so every
    probe is checked on content, finish, and finiteness.
    """

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
        if (
            not isinstance(value, (int, float))
            or value != value
            or value in (float("inf"), float("-inf"))
        ):
            return f"non-finite log-probability {value!r}"
    return None


def arithmetic_probe(*, logprobs: bool = True, **overrides) -> dict:
    payload = {
        "model": SPEC["model"]["served_name"],
        "messages": [{"role": "user", "content": "What is 17 * 19? Reply with only the integer."}],
        "max_tokens": 16384,
    }
    if logprobs:
        payload.update(logprobs=True, top_logprobs=3)
    payload.update(overrides)
    return payload


def _validate_arithmetic(api_url: str) -> None:
    # The template's default (max) and the lightest effort, one request per DP
    # rank and mode at once, so vLLM's DP balancer hands a probe to every rank.
    # It uses /v1/responses, the Chat UI's API, which has no token
    # log-probabilities (owner decision 2026-10-17): this probe checks the
    # exact answer and finish, and `verify.sh l1` reads the finite
    # log-probabilities from vLLM directly.
    variants = [variant for variant in ({}, {"reasoning_effort": "low"}) for _ in DP_RANK_GPU_IDS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(variants)) as pool:
        bodies = list(
            pool.map(
                lambda overrides: post_responses(
                    api_url,
                    arithmetic_probe(logprobs=False, **overrides),
                    timeout_s=900,
                ),
                variants,
            )
        )
    for overrides, body in zip(variants, bodies, strict=True):
        error = arithmetic_answer_error(body, logprobs=False)
        if error is not None:
            raise SystemExit(f"arithmetic probe {overrides or 'default'} failed: {error}")


_BASH_TOOL = {
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


def validate_tool_calling(api_url: str) -> None:
    payload = {
        "model": SPEC["model"]["served_name"],
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
        "tools": [_BASH_TOOL],
        "max_tokens": 16384,
    }
    try:
        body = post_responses(api_url, payload, timeout_s=900)
        choice = body["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        arguments = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    except (KeyError, IndexError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"tool-calling probe failed: {error}") from error
    if (
        choice.get("finish_reason") != "tool_calls"
        or not calls
        or calls[0]["function"].get("name") != "bash"
        or not isinstance(arguments, dict)
        or not isinstance(arguments.get("command"), str)
        or not arguments["command"]
    ):
        raise SystemExit(
            "tool-calling probe: the API did not return an executable bash tool "
            f"call (finish_reason={choice.get('finish_reason')!r}, tool_calls="
            f"{json.dumps(calls)[:300]})"
        )


# A 64x64 solid pure-red (255, 0, 0) PNG: readiness includes the vision encoder
# and an image span. The earlier (220, 40, 40) red was named "maroon",
# "crimson" or "salmon" in about 4 % of answers, directly from vLLM too.
PROBE_IMAGE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4"
    "FgYrsKZeS0BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc"
    "8OJg6Ln3AAAAAElFTkSuQmCC"
)


def image_request(text: str, *, max_tokens: int, images: int = 1) -> dict:
    image = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{PROBE_IMAGE_PNG_BASE64}"},
    }
    return {
        "model": SPEC["model"]["served_name"],
        "messages": [
            {"role": "user", "content": [*([image] * images), {"type": "text", "text": text}]}
        ],
        "max_tokens": max_tokens,
    }


def red_answer_error(body: dict) -> str | None:
    try:
        choice = body["choices"][0]
        content = choice["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return f"malformed response: {str(body)[:200]}"
    if choice.get("finish_reason") != "stop":
        return f"finish_reason is {choice.get('finish_reason')!r}"
    if not isinstance(content, str) or not re.search(r"\bred\b", content, re.IGNORECASE):
        return f"answer is {content!r}, expected red"
    return None


def validate_vision(api_url: str) -> None:
    payload = image_request(
        "What single color fills this image? Answer with one word.", max_tokens=16384
    )
    try:
        body = post_responses(api_url, payload, timeout_s=900)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"image probe failed: {error}") from error
    error = red_answer_error(body)
    if error is not None:
        raise SystemExit(f"image probe: {error}")


def validate_serving(api_url: str) -> None:
    """Readiness: pool state, exact finite answers (chat), a tool call and an image (Responses)."""

    validate_ready(api_url)
    _validate_arithmetic(api_url)
    validate_tool_calling(api_url)
    validate_vision(api_url)


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


def up() -> None:
    env = _compose_env()
    ui_bind = workspace.chat_ui_bind(WORKSPACE)
    ui_port = workspace.chat_ui_port(WORKSPACE)
    exposed = "0.0.0.0" in (env["API_BIND_ADDRESS"], ui_bind)
    host = public_host() if exposed else None
    ui_host = host if ui_bind == "0.0.0.0" else ui_bind
    api_host = host if env["API_BIND_ADDRESS"] == "0.0.0.0" else env["API_BIND_ADDRESS"]
    env["KAIRYU_WORKSPACE_URL"] = os.environ.get("WEBUI_URL", f"http://{ui_host}:{ui_port}")
    _preflight(env)
    _ensure_vllm_image(env)
    _ensure_model(env)
    workspace.fetch_models(WORKSPACE, Path(env["KAIRYU_WORKSPACE_MODELS"]))
    _compose(["up", "--build", "--detach", "--wait", "--wait-timeout", "7200"], env=env)
    validate_serving(api_check_url(env))
    ui_local = "127.0.0.1" if ui_bind == "0.0.0.0" else ui_bind
    workspace.setup(f"http://{ui_local}:{ui_port}", WORKSPACE)
    print("\nEnvironment is ready.")
    print(f"OpenAI API: http://{api_host}:{env['API_PORT']}/v1")
    print(f"Chat UI:    http://{ui_host}:{ui_port} (no authentication)")
    print(
        f"Chat model: {SPEC['model']['served_name']} "
        "(one TP2 x DP3 / EP6 replica on GPUs 0-5; text + image input; thinking max by default)"
    )
    print(
        "Reasoning effort: Chat Controls -> Valves -> Reasoning Effort "
        f"({'/'.join(WORKSPACE.effort_levels)})"
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
