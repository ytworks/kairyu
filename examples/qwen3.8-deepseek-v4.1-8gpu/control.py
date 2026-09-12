#!/usr/bin/env python3
"""Lifecycle for the unverified six-GPU V4.1 plus two-replica Qwen ensemble."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import secrets
import shutil
import socket
import stat
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


def _nvme_root() -> Path:
    configured = Path(os.environ.get("NVME_STORAGE_ROOT", SPEC["storage"]["root"]))
    if not configured.is_absolute():
        raise SystemExit("NVME_STORAGE_ROOT must be an absolute path below /mnt/nvme")
    root = configured.resolve()
    nvme = Path("/mnt/nvme")
    if root != nvme and nvme not in root.parents:
        raise SystemExit("NVME_STORAGE_ROOT must be /mnt/nvme or one of its descendants")
    return root


def _storage_paths(*, prepare: bool = False) -> dict[str, Path]:
    root = _nvme_root()
    environment = root / "model-volumes" / SPEC["environment"]
    paths = {
        "qwen_models": root / "model-volumes/qwen3.8-27b-1gpu/models",
        "deepseek_models": root / "model-volumes/deepseek-v4.1-flash-8gpu/models",
        "webui": environment / "webui-data",
        # Header-only FlashInfer fixes must never reuse previously compiled kernels.
        "deepseek_cache": environment / "compile-cache/deepseek-masked-kv-v1",
        "placement_log": environment / "placement-log",
    }
    for index in range(SPEC["allocation"]["tier1"]["replicas"]):
        paths[f"qwen_cache_{index}"] = environment / f"compile-cache/qwen-{index}"
    if prepare:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        minimum = int(SPEC["storage"]["minimum_free_gib"])
        free = shutil.disk_usage(root).free // (1024**3)
        if free < minimum:
            raise SystemExit(f"NVMe storage has {free} GiB free; {minimum} GiB is required")
    return paths


CONFIG_FILES = (
    "example.json",
    "compose.yaml",
    "kairyu.yaml",
    "auto-max.yaml",
    "router.json",
    "qwen3.8-chat.jinja",
    "requirements_budget.py",
    "control.py",
    "webui-reasoning-effort-filter.py",
    "vllm-sm120.Dockerfile",
    "patch_masked_kv.py",
    "../deepseek-v4.1-flash-8gpu/example.json",
    "../deepseek-v4.1-flash-8gpu/vllm-sm120.Dockerfile",
    "../deepseek-v4.1-flash-8gpu/patch_runtime.py",
    "../qwen3.8-deepseek-v4-8gpu/sandbox/Dockerfile",
    "../qwen3.8-deepseek-v4-8gpu/sandbox/runner.py",
    "../../kairyu/orchestration/conductor.py",
    "../../kairyu/dsl/spec.py",
    "../../kairyu/dsl/loader.py",
    "../../kairyu/entrypoints/server/sse_response.py",
    "../../kairyu/entrypoints/server/app.py",
    "../../kairyu/orchestration/orchestrator.py",
)


def _requirements_config_sha256() -> str:
    """Make mounted extractor changes visible to Compose's recreate decision."""
    digest = hashlib.sha256()
    for name in CONFIG_FILES:
        digest.update(name.encode() + b"\0" + (HERE / name).read_bytes() + b"\0")
    return digest.hexdigest()


def runtime_attestation() -> dict:
    """Reject stale or differently pinned containers; return no private env values."""
    project = SPEC["environment"].replace(".", "-")
    services = ("deepseek", "qwen-0", "qwen-1", "kairyu")
    names = {f"{project}-{service}-1": service for service in services}
    result = _run(["docker", "container", "inspect", *names], capture=True)
    containers = json.loads(result.stdout)
    digest = _requirements_config_sha256()
    report = {}
    for container in containers:
        name = container["Name"].lstrip("/")
        service = names.get(name)
        if service is None or not container.get("State", {}).get("Running"):
            raise SystemExit(f"expected running example container: {name}")
        env = dict(
            entry.split("=", 1) for entry in container["Config"].get("Env", []) if "=" in entry
        )
        if env.get("KAIRYU_REQUIREMENTS_CONFIG_SHA256") != digest:
            raise SystemExit(
                f"deployed configuration hash mismatch on {name}; run this checkout's run.sh up"
            )
        image = container["Image"]
        if service != "kairyu":
            expected = SPEC["vllm"]["deepseek" if service == "deepseek" else "qwen"]["image_id"]
            if image != expected:
                raise SystemExit(
                    f"deployed image ID mismatch on {name}: {image}, expected {expected}"
                )
        report[service] = {"container": name, "image_id": image}
    if set(report) != set(services):
        raise SystemExit("runtime attestation requires all four model/API containers")
    return {"config_sha256": digest, "containers": report}


def _compaction_secret(state: Path) -> str:
    """Persist one private key; adopt an already running example's key on upgrade."""
    key = "KAIRYU_RESPONSES_COMPACTION_SECRET"
    explicit = os.environ.get(key)
    if explicit:
        if len(explicit.encode()) < 32:
            raise SystemExit(f"{key} must contain at least 32 bytes")
        return explicit
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = state.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise SystemExit("example private state must be a directory owned by the launcher user")
    state.chmod(0o700)
    path = state / "compaction-secret"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise SystemExit("compaction secret must be a private regular file")
        os.fchmod(stream.fileno(), 0o600)
        value = stream.read()
        if not value:
            name = SPEC["environment"].replace(".", "-") + "-kairyu-1"
            existing = _run(["docker", "container", "inspect", name], capture=True, check=False)
            if existing.returncode and "No such" not in existing.stderr:
                raise SystemExit(
                    "cannot inspect existing API container; refusing to generate a replacement key"
                )
            if existing.returncode == 0:
                entries = json.loads(existing.stdout)[0]["Config"].get("Env", [])
                value = next(
                    (item.split("=", 1)[1] for item in entries if item.startswith(key + "=")), ""
                )
            if not value:
                value = secrets.token_hex(32)
            if len(value.encode()) < 32:
                raise SystemExit("deployed compaction secret is too short; refusing to rotate it")
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if len(value.encode()) < 32:
            raise SystemExit("stored compaction secret is invalid; refusing to rotate it")
        return value


def _compose_env(*, prepare: bool = False) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME"}
    }
    paths = _storage_paths(prepare=prepare)
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "KAIRYU_RESPONSES_COMPACTION_SECRET": (
                _compaction_secret(paths["webui"].parent / "private")
                if prepare
                else os.environ.get("KAIRYU_RESPONSES_COMPACTION_SECRET", "")
            ),
            "COMPOSE_PROJECT_NAME": SPEC["environment"].replace(".", "-"),
            "QWEN_MODEL_STORAGE_PATH": str(paths["qwen_models"]),
            "KAIRYU_REQUIREMENTS_CONFIG_SHA256": _requirements_config_sha256(),
            "DEEPSEEK_MODEL_STORAGE_PATH": str(paths["deepseek_models"]),
            "PLACEMENT_LOG_PATH": str(paths["placement_log"]),
            "WEBUI_STORAGE_PATH": str(paths["webui"]),
            "DEEPSEEK_CACHE_PATH": str(paths["deepseek_cache"]),
            "QWEN_VLLM_IMAGE": os.environ.get("QWEN_VLLM_IMAGE", SPEC["vllm"]["qwen"]["image"]),
            "DEEPSEEK_VLLM_IMAGE": os.environ.get(
                "DEEPSEEK_VLLM_IMAGE", SPEC["vllm"]["deepseek"]["image"]
            ),
            "OPEN_WEBUI_IMAGE": os.environ.get("OPEN_WEBUI_IMAGE", SPEC["webui"]["image"]),
            "API_BIND_ADDRESS": os.environ.get("API_BIND_ADDRESS", "0.0.0.0"),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "DEEPSEEK_L1_PORT": os.environ.get(
                "DEEPSEEK_L1_PORT", str(SPEC["deepseek_l1_loopback_port"])
            ),
            "CHAT_UI_PORT": os.environ.get("CHAT_UI_PORT", str(SPEC["webui"]["port"])),
            "CHAT_UI_BIND_ADDRESS": os.environ.get("CHAT_UI_BIND_ADDRESS", "0.0.0.0"),
            # Render-safe defaults for down/status/logs. `up` replaces these
            # with NUMA-local CPU sets discovered from the exact GPU inventory.
            "DEEPSEEK_CPUSET": os.environ.get("DEEPSEEK_CPUSET", "0"),
        }
    )
    for index in range(SPEC["allocation"]["tier1"]["replicas"]):
        env[f"QWEN_CACHE_{index}_PATH"] = str(paths[f"qwen_cache_{index}"])
        env[f"QWEN_{index}_CPUSET"] = os.environ.get(f"QWEN_{index}_CPUSET", "0")
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
        raise SystemExit(f"cannot determine NUMA affinity for GPU {pci_bus_id}: {error}") from error
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
    if sorted(rows) != list(range(expected["gpu_count"])):
        raise SystemExit("exactly eight contiguous GPU indices 0..7 are required")
    for index, row in rows.items():
        if row["name"] != expected["product"]:
            raise SystemExit(f"GPU {index} is {row['name']!r}; expected {expected['product']!r}")
        if row["memory_mib"] < expected["minimum_vram_mib"]:
            raise SystemExit(f"GPU {index} has insufficient VRAM")
        if row["compute_capability"] < expected["minimum_compute_capability"]:
            raise SystemExit(f"GPU {index} has insufficient compute capability")
    cpusets = {index: _numa_cpuset(str(row["pci_bus_id"])) for index, row in rows.items()}
    for index in range(SPEC["allocation"]["tier1"]["replicas"]):
        env[f"QWEN_{index}_CPUSET"] = cpusets[SPEC["allocation"]["tier1"]["gpu_ids"][index]]
    env["DEEPSEEK_CPUSET"] = ",".join(
        dict.fromkeys(cpusets[index] for index in SPEC["allocation"]["tier2"]["gpu_ids"])
    )
    print(
        f"hardware: 8 x {expected['product']} ({rows[0]['memory_mib']} MiB each); "
        "DeepSeek GPUs 0-5, Qwen GPUs 6-7",
        flush=True,
    )


def _image_id(image: str) -> str | None:
    result = _run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], capture=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _ensure_vllm_image(env: dict[str, str], env_key: str, source: dict) -> None:
    image = env[env_key]
    actual = _image_id(image)
    if actual is None:
        if image != source["image"]:
            raise SystemExit(f"{env_key} does not exist locally: {image}")
        if source.get("distribution") == "upstream":
            _run(["docker", "pull", image])
        else:
            parent_example = source.get("parent_runtime_example")
            if parent_example:
                parent_path = (HERE / parent_example).resolve()
                parent = json.loads(parent_path.read_text())["vllm"]
                if parent["image"] != source["base_image"]:
                    raise SystemExit("parent runtime image does not match the overlay base")
                parent["dockerfile"] = str(parent_path.parent / parent["dockerfile"])
                _ensure_vllm_image({**env, env_key: parent["image"]}, env_key, parent)
            dockerfile = (HERE / source["dockerfile"]).resolve()
            _run(
                [
                    "docker",
                    "build",
                    *([] if parent_example else ["--pull"]),
                    "--file",
                    str(dockerfile),
                    "--build-arg",
                    f"VLLM_BASE_IMAGE={source['base_image']}",
                    "--build-arg",
                    f"FLASHINFER_REVISION={source['flashinfer_revision']}",
                    "--tag",
                    image,
                    "--label",
                    f"org.opencontainers.image.revision={source['source_revision']}",
                    str(dockerfile.parent),
                ]
            )
        actual = _image_id(image)
    if actual != source["image_id"]:
        raise SystemExit(
            f"vLLM image {image} has ID {actual}; expected {source['image_id']}. "
            "Attest a rebuilt runtime and update example.json plus kairyu.yaml together."
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


def _download_model(image: str, mount: str, model: dict) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        "0:0",
        "--entrypoint",
        "python3",
        "--volume",
        f"{mount}:/models",
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


def _ensure_models(env: dict[str, str]) -> None:
    _download_model(env["QWEN_VLLM_IMAGE"], env["QWEN_MODEL_STORAGE_PATH"], SPEC["models"]["tier1"])
    _download_model(
        env["DEEPSEEK_VLLM_IMAGE"], env["DEEPSEEK_MODEL_STORAGE_PATH"], SPEC["models"]["tier2"]
    )


def _prepare_deepseek_cache(env: dict[str, str]) -> None:
    # The pinned V4.1 overlay is root-run; PR595 used a different non-root
    # DeepSeek image, whose cache ownership policy does not apply here.
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "0:0",
            "--entrypoint",
            "sh",
            "--volume",
            f"{env['DEEPSEEK_CACHE_PATH']}:/cache",
            env["DEEPSEEK_VLLM_IMAGE"],
            "-c",
            "mkdir -p /cache/torchinductor /cache/triton /cache/tilelang/tmp",
        ]
    )


def _json_url(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


_CHAT_UI_EFFORT_FILTER_ID = "reasoning_effort"
_CHAT_UI_EFFORT_LEVELS = ["default", "low", "high", "max"]


def _webui_api(
    ui_url: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict | None = None,
    method: str | None = None,
):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{ui_url}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method=method or ("POST" if payload is not None else "GET"),
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _provision_chat_ui_effort_selector(ui_url: str) -> None:
    """Install the Reasoning Effort dropdown into the pinned Open WebUI.

    The stock v0.11.0 Advanced Params control is a free-text field; the
    product knob (DTO-D6) must be selectable. A global filter whose
    enum-typed user valve renders as a dropdown in Chat Controls forwards
    the selection as the OpenAI-compatible ``reasoning_effort`` body field.
    """

    filter_source = (HERE / "webui-reasoning-effort-filter.py").read_text(encoding="utf-8")
    try:
        signin = _webui_api(ui_url, "/api/v1/auths/signin", payload={"email": "", "password": ""})
        if not isinstance(signin, dict) or not signin.get("token"):
            raise SystemExit("Chat UI signin did not return an auth-disabled session token")
        if signin.get("role") != "admin":
            raise SystemExit("Chat UI auth-disabled session must be the admin user")
        token = signin["token"]
        listed = _webui_api(ui_url, "/api/v1/functions/", token=token)
        existing = next((row for row in listed if row.get("id") == _CHAT_UI_EFFORT_FILTER_ID), None)
        body = {
            "id": _CHAT_UI_EFFORT_FILTER_ID,
            "name": "Reasoning Effort",
            "content": filter_source,
            "meta": {"description": "Select the Kairyu reasoning effort from a dropdown."},
        }
        if existing is None:
            state = _webui_api(ui_url, "/api/v1/functions/create", token=token, payload=body)
        else:
            state = _webui_api(
                ui_url,
                f"/api/v1/functions/id/{_CHAT_UI_EFFORT_FILTER_ID}/update",
                token=token,
                payload=body,
            )
            # /update keeps the stored activation flags; carry them over.
            state = {**existing, **(state or {})}
        # The /toggle endpoints FLIP state, so only call them while the flag
        # is off — calling unconditionally would deactivate on every re-up.
        if not state.get("is_active"):
            state = _webui_api(
                ui_url,
                f"/api/v1/functions/id/{_CHAT_UI_EFFORT_FILTER_ID}/toggle",
                token=token,
                payload={},
            )
        if not state.get("is_global"):
            state = _webui_api(
                ui_url,
                f"/api/v1/functions/id/{_CHAT_UI_EFFORT_FILTER_ID}/toggle/global",
                token=token,
                payload={},
            )
        if not (state.get("is_active") and state.get("is_global")):
            raise SystemExit("Chat UI effort selector could not be activated globally")
        spec = _webui_api(
            ui_url,
            f"/api/v1/functions/id/{_CHAT_UI_EFFORT_FILTER_ID}/valves/user/spec",
            token=token,
        )
        enum = spec.get("properties", {}).get("reasoning_effort", {}).get("enum")
    except (KeyError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Chat UI effort selector provisioning failed: {error}") from error
    if enum != _CHAT_UI_EFFORT_LEVELS:
        raise SystemExit(
            f"Chat UI effort selector must expose exactly {_CHAT_UI_EFFORT_LEVELS!r}, got {enum!r}"
        )


def _post_json_url(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def _validate_embedding_smoke(payload: dict) -> None:
    embedding = SPEC["embedding"]
    model = embedding["served_name"]
    if payload.get("model") != model:
        raise SystemExit(
            "Kairyu embedding response has the wrong model identity: "
            f"expected {model!r}, got {payload.get('model')!r}"
        )
    data = payload.get("data")
    if not isinstance(data, list) or [
        row.get("index") if isinstance(row, dict) else None for row in data
    ] != [0, 1]:
        raise SystemExit("Kairyu embedding response must preserve indices [0, 1]")
    dimensions = embedding["dimensions"]
    for row in data:
        vector = row.get("embedding")
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise SystemExit(f"Kairyu embedding response vectors must have {dimensions} dimensions")
        if not all(type(value) in {int, float} and math.isfinite(value) for value in vector):
            raise SystemExit("Kairyu embedding response vectors must contain finite numbers")
    usage = payload.get("usage")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    if (
        type(prompt_tokens) is not int
        or prompt_tokens < 1
        or type(total_tokens) is not int
        or total_tokens != prompt_tokens
    ):
        raise SystemExit("Kairyu embedding response must report positive exact usage")


def _validate_ready(api_url: str, tokenizer_url: str) -> None:
    try:
        ready = _json_url(f"{api_url}/readyz")
        models = {row["id"] for row in _json_url(f"{api_url}/v1/models")["data"]}
        routing = _json_url(f"{api_url}/routing")["models"]
        embedding_model = SPEC["embedding"]["served_name"]
        embedding_response = _post_json_url(
            f"{api_url}/v1/embeddings",
            {
                "model": embedding_model,
                "input": ["kairyu readiness probe", "two-input contract"],
                "encoding_format": "float",
            },
        )
        token_count = _post_json_url(
            tokenizer_url,
            {"model": "deepseek-v4.1-flash", "prompt": "kairyu"},
        )["count"]
    except (KeyError, OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Kairyu readiness evidence is incomplete: {error}") from error
    product_model = SPEC["orchestration"]["auto_max_model"]
    expected_models = {product_model, embedding_model}
    if ready.get("status") != "ready" or models != expected_models:
        raise SystemExit(
            "Kairyu public model inventory must be exactly "
            f"{sorted(expected_models)!r}, got {sorted(models)!r}"
        )
    _validate_embedding_smoke(embedding_response)
    if type(token_count) is not int or token_count < 1:
        raise SystemExit("DeepSeek public-output tokenizer oracle is not ready")
    if set(routing) != {product_model}:
        raise SystemExit(f"Kairyu product routing inventory is not isolated: {sorted(routing)}")
    policy = routing[product_model]
    orchestration = SPEC["orchestration"]
    expected_roles = list(orchestration["roles"])
    if [role.get("name") for role in policy.get("roles", ())] != expected_roles:
        raise SystemExit(
            "Kairyu L2 does not report the required "
            f"{len(expected_roles)}-role dual-track product DAG"
        )
    # DTO-D13: the dual-track DAG is the primary profile behind a Qwen route
    # judge that selects among four single-role direct routes and the
    # ensemble; a missing profile or judge means the wrong policy is live.
    expected_profiles = {name: list(roles) for name, roles in orchestration["profiles"].items()}
    served_profiles = {
        name: [role.get("name") for role in roles]
        for name, roles in (policy.get("profiles") or {}).items()
    }
    if served_profiles != expected_profiles:
        raise SystemExit(
            "Kairyu L2 does not report the required direct-route profiles "
            f"{sorted(expected_profiles)!r}, got {sorted(served_profiles)!r}"
        )
    expected_sampling = orchestration["direct_route_sampling"]
    served_sampling = {
        name: {
            key: value
            for key, value in (roles[0].get("sampling") or {}).items()
            if value not in (None, [], {})
        }
        for name, roles in (policy.get("profiles") or {}).items()
    }
    if served_sampling != expected_sampling:
        raise SystemExit("Kairyu L2 does not report the required direct-route sampling policy")
    expected_judge = orchestration["profile_judge"]
    judge = policy.get("profile_judge") or {}
    served_choices = [
        {"label": choice.get("label"), "profile": choice.get("profile")}
        for choice in judge.get("choices", ())
    ]
    if (
        judge.get("worker") != expected_judge["worker"]
        or judge.get("fallback") != expected_judge["fallback"]
        or served_choices != expected_judge["choices"]
    ):
        raise SystemExit(
            "Kairyu product policy must judge routes on the Qwen worker with the "
            f"{len(expected_judge['choices'])} configured choices"
        )
    if policy.get("stream_head") != orchestration["stream_head"]:
        raise SystemExit("Kairyu product policy must stream the head role publicly")
    if policy.get("moa_samples") != 0:
        raise SystemExit("Kairyu product policy must use the explicit DAG, not MoA")
    if policy.get("budget", {}).get("max_steps") != orchestration["max_steps"]:
        raise SystemExit(f"Kairyu product policy max_steps must be {orchestration['max_steps']}")
    expected_refinements = orchestration["product_max_refinements"]
    if policy.get("budget", {}).get("max_refine_depth") != expected_refinements:
        raise SystemExit(f"Kairyu product policy max_refine_depth must be {expected_refinements}")
    if policy.get("expose_intermediate_outputs") is not True:
        raise SystemExit("Kairyu product policy must expose separate intermediate output")
    configured = policy.get("configured_engines", {})
    if configured.get("tier1", {}).get("model") != "qwen3.8-27b":
        raise SystemExit("Kairyu Tier1 L2 worker is not bound to the Qwen L1 pool")
    if configured.get("tier2", {}).get("model") != "deepseek-v4.1-flash-thinking":
        raise SystemExit("Kairyu Tier2 L2 worker is not bound to the thinking DeepSeek L1 pool")
    if configured.get("tier2-direct", {}).get("model") != "deepseek-v4.1-flash":
        raise SystemExit(
            "Kairyu tier2-direct L2 worker is not bound to the non-thinking DeepSeek L1 pool"
        )


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
        raise SystemExit(
            "cannot discover an externally reachable Chat UI host; set PUBLIC_HOST"
        ) from error
    if detected == "0.0.0.0" or detected.startswith("127."):
        raise SystemExit("cannot discover an externally reachable Chat UI host; set PUBLIC_HOST")
    return detected


def up() -> None:
    env = _compose_env(prepare=True)
    ui_host = _public_ui_host()
    env["WEBUI_URL"] = os.environ.get("WEBUI_URL", f"http://{ui_host}:{env['CHAT_UI_PORT']}")
    _preflight(env)
    _ensure_vllm_image(env, "QWEN_VLLM_IMAGE", SPEC["vllm"]["qwen"])
    _ensure_vllm_image(env, "DEEPSEEK_VLLM_IMAGE", SPEC["vllm"]["deepseek"])
    _ensure_models(env)
    _prepare_deepseek_cache(env)
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
    validation_api_host = (
        "127.0.0.1" if env["API_BIND_ADDRESS"] == "0.0.0.0" else env["API_BIND_ADDRESS"]
    )
    api_url = f"http://{validation_api_host}:{env['API_PORT']}"
    advertised_api_host = (
        ui_host if env["API_BIND_ADDRESS"] == "0.0.0.0" else env["API_BIND_ADDRESS"]
    )
    tokenizer_url = f"http://127.0.0.1:{env['DEEPSEEK_L1_PORT']}/tokenize"
    _validate_ready(api_url, tokenizer_url)
    validation_ui_host = (
        "127.0.0.1" if env["CHAT_UI_BIND_ADDRESS"] == "0.0.0.0" else env["CHAT_UI_BIND_ADDRESS"]
    )
    _provision_chat_ui_effort_selector(f"http://{validation_ui_host}:{env['CHAT_UI_PORT']}")
    print("\nEnvironment is ready.")
    print(f"OpenAI API: http://{advertised_api_host}:{env['API_PORT']}/v1")
    print(f"Chat UI:    http://{ui_host}:{env['CHAT_UI_PORT']} (no authentication)")
    print("Chat model:      kairyu-auto-max (the only Chat UI model)")
    print("Embedding model: embed-small")
    print("Reasoning effort: Chat Controls -> Valves -> Reasoning Effort (default/low/high/max)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", choices=("up", "down", "status", "logs", "config"), default="up"
    )
    args = parser.parse_args()
    if args.action == "up":
        up()
    elif args.action == "down":
        _compose(["down"])
    elif args.action == "status":
        _compose(["ps"])
    elif args.action == "config":
        _compose(["config"])
    else:
        _compose(["logs", "--follow", "--tail", "200"])


if __name__ == "__main__":
    main()
