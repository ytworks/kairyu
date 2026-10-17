#!/usr/bin/env python3
"""One-command lifecycle for DeepSeek-V4-Flash-0731 as two TP4+EP4 replicas on eight GPUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
ROOT = HERE.parents[1]
REPLICAS = int(SPEC["allocation"]["replicas"])
REPLICA_GPU_IDS: list[list[int]] = [list(group) for group in SPEC["allocation"]["replica_gpu_ids"]]


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


def _storage_paths() -> dict[str, Path]:
    root = _nvme_root()
    environment = root / "model-volumes" / SPEC["environment"]
    # The checkpoint volume is shared with the single-model eight-GPU example
    # (and the tiered example) so one attested download serves all of them.
    paths = {
        "models": root / "model-volumes" / SPEC["storage"]["shared_model_environment"],
        "webui": environment / "webui-data",
        "placement_log": environment / "placement-log",
    }
    paths.update(
        {
            f"cache_{index}": environment / f"compile-cache/deepseek-{index}"
            for index in range(REPLICAS)
        }
    )
    for path in paths.values():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(f"cannot prepare NVMe storage {path}: {error}") from error
    free_gib = shutil.disk_usage(root).free // (1024**3)
    minimum = int(SPEC["storage"]["minimum_free_gib"])
    if free_gib < minimum:
        raise SystemExit(f"NVMe storage has {free_gib} GiB free; {minimum} GiB is required")
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
            "COMPOSE_PROJECT_NAME": SPEC["environment"].replace(".", "-"),
            "DEEPSEEK_MODEL_VOLUME": os.environ.get(
                "DEEPSEEK_MODEL_VOLUME", SPEC["storage"]["deepseek_model_volume"]
            ),
            "WEBUI_STORAGE_PATH": str(paths["webui"]),
            "PLACEMENT_LOG_PATH": str(paths["placement_log"]),
            "DEEPSEEK_VLLM_IMAGE": os.environ.get("DEEPSEEK_VLLM_IMAGE", SPEC["vllm"]["image"]),
            "DEEPSEEK_SERVED_VLLM_IMAGE": SPEC["vllm"]["effort_overlay"]["image"],
            "OPEN_WEBUI_IMAGE": os.environ.get("OPEN_WEBUI_IMAGE", SPEC["webui"]["image"]),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "CHAT_UI_PORT": os.environ.get("CHAT_UI_PORT", str(SPEC["webui"]["port"])),
            "CHAT_UI_BIND_ADDRESS": os.environ.get("CHAT_UI_BIND_ADDRESS", "127.0.0.1"),
        }
    )
    for index in range(REPLICAS):
        env[f"DEEPSEEK_CACHE_{index}_PATH"] = str(paths[f"cache_{index}"])
        # Render-safe default for down/status/logs; `up` replaces it with the
        # union of the replica's GPUs' NUMA-local CPU sets.
        env[f"DEEPSEEK_{index}_CPUSET"] = os.environ.get(f"DEEPSEEK_{index}_CPUSET", "0")
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
    expected = SPEC["hardware"]
    if sorted(rows) != list(SPEC["allocation"]["gpu_ids"]):
        raise SystemExit(
            f"exactly {expected['gpu_count']} contiguous GPU indices "
            f"0..{expected['gpu_count'] - 1} are required"
        )
    for index, row in rows.items():
        if row["name"] != expected["product"]:
            raise SystemExit(
                f"GPU {index} is {row['name']!r}; expected {expected['product']!r}"
            )
        if row["memory_mib"] < expected["minimum_vram_mib"]:
            raise SystemExit(f"GPU {index} has insufficient VRAM")
        if row["compute_capability"] < expected["minimum_compute_capability"]:
            raise SystemExit(f"GPU {index} has insufficient compute capability")
    cpusets = {index: _numa_cpuset(str(row["pci_bus_id"])) for index, row in rows.items()}
    for replica, group in enumerate(REPLICA_GPU_IDS):
        env[f"DEEPSEEK_{replica}_CPUSET"] = ",".join(
            dict.fromkeys(cpusets[index] for index in group)
        )
    layout = "; ".join(
        f"replica {replica} on GPUs {group[0]}-{group[-1]}"
        for replica, group in enumerate(REPLICA_GPU_IDS)
    )
    print(
        f"hardware: {len(rows)} x {expected['product']} ({rows[0]['memory_mib']} MiB each); "
        f"{layout}",
        flush=True,
    )


def _image_id(image: str) -> str | None:
    inspected = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture=True,
        check=False,
    )
    return inspected.stdout.strip() if inspected.returncode == 0 else None


def _image_exists(image: str) -> bool:
    return _run(["docker", "image", "inspect", image], capture=True, check=False).returncode == 0


def _ensure_vllm_image(env: dict[str, str]) -> None:
    image = env["DEEPSEEK_VLLM_IMAGE"]
    if _image_exists(image):
        return
    source = SPEC["vllm"]
    if image != source["image"]:
        raise SystemExit(f"DEEPSEEK_VLLM_IMAGE does not exist locally: {image}")
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
            f"{source['source_repository']}#{source['source_revision']}",
        ]
    )


def _ensure_effort_overlay(env: dict[str, str]) -> None:
    """Build the served image: the pinned build plus the xhigh->max effort patch.

    Kairyu sends an L3 max request to vLLM's /v1/responses as xhigh (m11 D4
    amendment 2026-10-17); patch_deepseek_v4_effort.py renders it as max. The
    layer cache makes an unchanged rebuild instant, and an edited patch is
    never served stale. The served image is attested by its own recorded ID.
    """

    overlay = SPEC["vllm"]["effort_overlay"]
    _run(
        [
            "docker",
            "build",
            "--file",
            str(HERE / overlay["dockerfile"]),
            "--build-arg",
            f"VLLM_BASE_IMAGE={env['DEEPSEEK_VLLM_IMAGE']}",
            "--tag",
            env["DEEPSEEK_SERVED_VLLM_IMAGE"],
            str(HERE),
        ]
    )
    image = env["DEEPSEEK_SERVED_VLLM_IMAGE"]
    actual = _image_id(image)
    if os.environ.get("DEEPSEEK_ALLOW_UNPINNED_IMAGE") == "1":
        print(f"WARNING: serving unpinned image {image} ({actual}); not evidence", flush=True)
        return
    if overlay["image_id"] is None:
        raise SystemExit(
            f"vLLM image {image} ({actual}) has no recorded image ID: it was rebuilt for "
            "the Responses xhigh->max effort edit and awaits GPU verification; record this "
            "ID as example.json image_id and kairyu.yaml container_image_digest, or serve it "
            "anyway (not evidence) with DEEPSEEK_ALLOW_UNPINNED_IMAGE=1"
        )
    if actual != overlay["image_id"]:
        raise SystemExit(
            f"vLLM image {image} has ID {actual}; example.json and kairyu.yaml pin "
            f"{overlay['image_id']} (update both to this build before serving)"
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


def _ensure_model_volume(env: dict[str, str]) -> str:
    """Create or check the NVMe-bind Docker volume shared with the sibling examples."""

    name = env["DEEPSEEK_MODEL_VOLUME"]
    desired = str(_storage_paths()["models"])
    inspected = _run(["docker", "volume", "inspect", name], capture=True, check=False)
    if inspected.returncode:
        _run(
            [
                "docker",
                "volume",
                "create",
                "--driver",
                "local",
                "--opt",
                "type=none",
                "--opt",
                "o=bind",
                "--opt",
                f"device={desired}",
                name,
            ],
            capture=True,
        )
        inspected = _run(["docker", "volume", "inspect", name], capture=True)
    payload = json.loads(inspected.stdout)
    options = payload[0].get("Options") if len(payload) == 1 else None
    expected = {"device": desired, "o": "bind", "type": "none"}
    if options != expected:
        raise SystemExit(
            f"existing volume {name} is not the expected NVMe bind: "
            f"expected {expected}, found {options}"
        )
    return name


def _ensure_model(env: dict[str, str]) -> None:
    volume = _ensure_model_volume(env)
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
            env["DEEPSEEK_VLLM_IMAGE"],
            "-c",
            _MODEL_PROGRAM,
            model["repo"],
            model["revision"],
            model["slug"],
            model["tree_sha256"],
        ]
    )
    _run(command)


def _json_url(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def _text_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def _healthy_replicas(metrics: str, pool: str) -> int | None:
    pattern = re.compile(
        r'^kairyu_pool_replicas\{pool="' + re.escape(pool) + r'",state="healthy"\} (\S+)$',
        re.MULTILINE,
    )
    match = pattern.search(metrics)
    return int(float(match.group(1))) if match else None


def _validate_ready(api_url: str) -> None:
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
    if healthy != REPLICAS:
        raise SystemExit(
            f"Kairyu pool {served!r} must report {REPLICAS} healthy replicas, got {healthy!r}"
        )


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
            text = message["reasoning_content"]
            items.append(
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": text}],
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
            reasoning += [part.get("text", "") for part in parts]
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


# One fail-closed tool-calling probe (PR #584 review): `run.sh up` must not
# report a ready environment whose API cannot emit OpenAI tool calls. It goes
# through /v1/responses, the API the Chat UI uses.
_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run."}
            },
            "required": ["command"],
        },
    },
}


def _validate_tool_calling(api_url: str) -> None:
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
        "max_tokens": 2048,
    }
    request = urllib.request.Request(
        f"{api_url}/v1/responses",
        data=json.dumps(responses_request(payload)).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = chat_view(json.loads(response.read()))
        choice = body["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        arguments = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    except (KeyError, IndexError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"tool-calling probe failed: {error}") from error
    if (
        choice.get("finish_reason") != "tool_calls"
        or calls[0]["function"].get("name") != "bash"
        or not isinstance(arguments.get("command"), str)
    ):
        raise SystemExit(
            "tool-calling probe: the API did not return an executable bash tool "
            f"call (finish_reason={choice.get('finish_reason')!r}, tool_calls="
            f"{json.dumps(calls)[:300]})"
        )


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
    env = _compose_env()
    _preflight(env)
    _ensure_vllm_image(env)
    _ensure_effort_overlay(env)
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
            "7200",
        ],
        env=env,
    )
    api_url = f"http://127.0.0.1:{env['API_PORT']}"
    _validate_ready(api_url)
    _validate_tool_calling(api_url)
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
    print(
        f"Chat model: {SPEC['model']['served_name']} "
        f"({REPLICAS} TP4+EP4 replicas, four GPUs each)"
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
